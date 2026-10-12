"""Shared fixes for the new revision entry; original training files are untouched."""
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def image_map(paths):
    paths = paths if isinstance(paths, list) else [paths]
    output = {}
    for item in paths:
        p = Path(item)
        if p.is_dir():
            files = sorted(p.rglob('*'))
        elif p.is_file() and p.suffix.lower() == '.txt':
            files = [Path(x) if Path(x).is_absolute() else p.parent / x
                     for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        else:
            raise FileNotFoundError(p)
        for f in files:
            if not f.is_file() or f.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
                continue
            if f.stem in output:
                raise ValueError('Duplicate image stem, use globally unique IDs: ' + f.stem)
            output[f.stem] = f.resolve()
    if not output:
        raise ValueError('No images: ' + str(paths))
    return output


def label_path(image, suffix):
    parts = list(Path(image).parts)
    if 'images' not in parts:
        raise ValueError('Expected an images directory in ' + str(image))
    parts[parts.index('images')] = 'labels'
    return Path(*parts).with_suffix(suffix)


class PairedDataset(Dataset):
    """Read all real/fake roots; geometrically transform image, boxes, mask together.

    Target datasets never open GT labels or masks. Source masks come from the
    segmentation image tree, matched by unique image ID rather than batch index.
    """
    def __init__(self, real, fake, size, hyp, masks=None, nc=None, nseg=None):
        self.real, fake_map = image_map(real), image_map(fake)
        if self.real.keys() != fake_map.keys():
            missing = sorted(self.real.keys() - fake_map.keys())[:5]
            extra = sorted(fake_map.keys() - self.real.keys())[:5]
            raise ValueError('Real/fake ID mismatch; missing=%s extra=%s' % (missing, extra))
        self.ids = sorted(self.real)
        self.img_files = [str(self.real[k]) for k in self.ids]
        self.img_files_fake = [str(fake_map[k]) for k in self.ids]
        self.size, self.hyp, self.nseg = size, hyp, nseg
        self.n, self.indices = len(self.ids), list(range(len(self.ids)))
        self.labels, self.masks = [], []
        mask_map = image_map(masks) if masks is not None else None
        if mask_map is not None and self.real.keys() != mask_map.keys():
            raise ValueError('Source detection/segmentation image IDs must be identical')
        if hyp.get('mixup', 0) != 0:
            raise ValueError('Revision mask pipeline requires mixup=0')
        for k in self.ids:
            if mask_map is None:
                self.labels.append(np.empty((0, 5), dtype=np.float32))
                continue
            lp, mp = label_path(self.real[k], '.txt'), label_path(mask_map[k], '.png')
            if not lp.is_file() or not mp.is_file():
                raise FileNotFoundError('Required source labels: %s, %s' % (lp, mp))
            rows = [line.split() for line in lp.read_text().splitlines() if line.strip()]
            a = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
            if (not np.isfinite(a).all() or (a[:, 1:] < 0).any() or
                    (a[:, 1:] > 1).any() or (a[:, 0] != a[:, 0].astype(int)).any() or
                    (a[:, 0] < 0).any() or (nc is not None and (a[:, 0] >= nc).any())):
                raise ValueError('Invalid source detection labels: ' + str(lp))
            self.labels.append(a)
            self.masks.append(mp)

    def __len__(self):
        return self.n

    def load(self, index):
        a = cv2.imread(self.img_files[index])
        b = cv2.imread(self.img_files_fake[index])
        if a is None or b is None or a.shape != b.shape:
            raise ValueError('Real/fake must decode to identical geometry: ' + self.img_files[index])
        h0, w0 = a.shape[:2]
        r = self.size / max(h0, w0)
        w, h = max(1, int(w0*r)), max(1, int(h0*r))
        a, b = cv2.resize(a, (w, h)), cv2.resize(b, (w, h))
        mask = None
        if self.masks:
            # Preserve indexed PNG class IDs; do not convert palette colours to grey.
            mask = np.asarray(Image.open(self.masks[index]))
            if mask.ndim != 2 or mask.shape != (h0, w0):
                raise ValueError('Mask must contain class IDs and match image geometry: ' + str(self.masks[index]))
            if (mask < 0).any() or (mask >= self.nseg).any():
                raise ValueError('Mask has IDs outside [0,n_segcls); map void/255 explicitly first: ' + str(self.masks[index]))
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        labels = self.labels[index].copy()
        if len(labels):
            xywh = labels[:, 1:].copy()
            labels[:, 1] = (xywh[:, 0] - xywh[:, 2]/2)*w
            labels[:, 2] = (xywh[:, 1] - xywh[:, 3]/2)*h
            labels[:, 3] = (xywh[:, 0] + xywh[:, 2]/2)*w
            labels[:, 4] = (xywh[:, 1] + xywh[:, 3]/2)*h
        return a, b, labels, mask

    def __getitem__(self, index):
        from utils.datasets import augment_hsv, box_candidates
        s, hyp = self.size, self.hyp
        use_mosaic = random.random() < hyp.get('mosaic', 0)
        if use_mosaic:
            yc, xc = [int(random.uniform(s/2, 1.5*s)) for _ in range(2)]
            a = np.full((s*2, s*2, 3), 114, dtype=np.uint8)
            b = a.copy()
            mask = np.zeros((s*2, s*2), dtype=np.uint8) if self.masks else None
            labels = []
            for i, idx in enumerate([index] + random.choices(self.indices, k=3)):
                ai, bi, li, mi = self.load(idx)
                h, w = ai.shape[:2]
                if i == 0:
                    x1,y1,x2,y2 = max(xc-w,0),max(yc-h,0),xc,yc
                    u1,v1,u2,v2 = w-(x2-x1),h-(y2-y1),w,h
                elif i == 1:
                    x1,y1,x2,y2 = xc,max(yc-h,0),min(xc+w,s*2),yc
                    u1,v1,u2,v2 = 0,h-(y2-y1),x2-x1,h
                elif i == 2:
                    x1,y1,x2,y2 = max(xc-w,0),yc,xc,min(yc+h,s*2)
                    u1,v1,u2,v2 = w-(x2-x1),0,w,y2-y1
                else:
                    x1,y1,x2,y2 = xc,yc,min(xc+w,s*2),min(yc+h,s*2)
                    u1,v1,u2,v2 = 0,0,x2-x1,y2-y1
                a[y1:y2,x1:x2], b[y1:y2,x1:x2] = ai[v1:v2,u1:u2], bi[v1:v2,u1:u2]
                if mask is not None:
                    mask[y1:y2,x1:x2] = mi[v1:v2,u1:u2]
                li[:, [1,3]] += x1-u1
                li[:, [2,4]] += y1-v1
                labels.append(li)
            labels = np.concatenate(labels, axis=0)
            labels[:, 1:] = labels[:, 1:].clip(0, 2*s)
        else:
            a,b,labels,mask = self.load(index)
            h,w = a.shape[:2]
            top,left = (s-h)//2,(s-w)//2
            pad = (top,s-h-top,left,s-w-left)
            a = cv2.copyMakeBorder(a,*pad,cv2.BORDER_CONSTANT,value=(114,114,114))
            b = cv2.copyMakeBorder(b,*pad,cv2.BORDER_CONSTANT,value=(114,114,114))
            if mask is not None:
                mask = cv2.copyMakeBorder(mask,*pad,cv2.BORDER_CONSTANT,value=0)
            labels[:, [1,3]] += left
            labels[:, [2,4]] += top
        # Shared projective transform, including mosaic crop, with nearest mask interpolation.
        c = np.eye(3); c[0,2],c[1,2] = -a.shape[1]/2,-a.shape[0]/2
        p = np.eye(3)
        p[2,0],p[2,1] = [random.uniform(-hyp['perspective'],hyp['perspective']) for _ in range(2)]
        r = np.eye(3)
        angle,scale = random.uniform(-hyp['degrees'],hyp['degrees']),random.uniform(1-hyp['scale'],1+hyp['scale'])
        r[:2] = cv2.getRotationMatrix2D((0,0),angle,scale)
        sh = np.eye(3)
        sh[0,1],sh[1,0] = [math.tan(random.uniform(-hyp['shear'],hyp['shear'])*math.pi/180) for _ in range(2)]
        tr = np.eye(3)
        tr[0,2],tr[1,2] = [random.uniform(.5-hyp['translate'],.5+hyp['translate'])*s for _ in range(2)]
        matrix = tr @ sh @ r @ p @ c
        a = cv2.warpPerspective(a,matrix,(s,s),borderValue=(114,114,114))
        b = cv2.warpPerspective(b,matrix,(s,s),borderValue=(114,114,114))
        if mask is not None:
            mask = cv2.warpPerspective(mask,matrix,(s,s),flags=cv2.INTER_NEAREST,borderValue=0)
        if len(labels):
            old = labels[:,1:].copy()
            xy = np.ones((len(labels)*4,3))
            xy[:,:2] = labels[:,[1,2,3,4,1,4,3,2]].reshape(-1,2)
            xy = xy @ matrix.T
            xy = (xy[:,:2]/xy[:,2:]).reshape(-1,8)
            x,y = xy[:,[0,2,4,6]],xy[:,[1,3,5,7]]
            new = np.stack([x.min(1),y.min(1),x.max(1),y.max(1)],axis=1).clip(0,s)
            keep = box_candidates(old.T*scale,new.T,area_thr=.1)
            labels = labels[keep]; labels[:,1:] = new[keep]
        augment_hsv(a,b,hyp['hsv_h'],hyp['hsv_s'],hyp['hsv_v'])
        if random.random() < hyp['flipud']:
            a,b = np.flipud(a),np.flipud(b)
            if mask is not None: mask = np.flipud(mask)
            labels[:,[2,4]] = s-labels[:,[4,2]]
        if random.random() < hyp['fliplr']:
            a,b = np.fliplr(a),np.fliplr(b)
            if mask is not None: mask = np.fliplr(mask)
            labels[:,[1,3]] = s-labels[:,[3,1]]
        if len(labels):
            xyxy = labels[:,1:].copy()
            labels[:,1:3] = (xyxy[:,:2]+xyxy[:,2:])/2/s
            labels[:,3:5] = (xyxy[:,2:]-xyxy[:,:2])/s
        target = torch.zeros((len(labels),6),dtype=torch.float32)
        if len(labels): target[:,1:] = torch.from_numpy(labels)
        tensor = lambda im: torch.from_numpy(np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)))
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long() if mask is not None else None
        return tensor(a),tensor(b),target,self.img_files[index],self.img_files_fake[index],None,mask

    @staticmethod
    def collate(batch):
        a,b,t,pa,pb,sh,m = zip(*batch)
        for i,row in enumerate(t): row[:,0] = i
        return torch.stack(a),torch.stack(b),torch.cat(t),pa,pb,sh,torch.stack(m) if m[0] is not None else None


def create_revision_loader(real, fake, size, batch, hyp, seed, workers, masks=None, nc=None, nseg=None):
    ds = PairedDataset(real,fake,size,hyp,masks,nc,nseg)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds,batch_size=batch,shuffle=True,num_workers=min(workers,batch),
                        pin_memory=True,worker_init_fn=seed_worker,generator=generator,collate_fn=ds.collate)
    return loader,ds


def safe_dynamic_k(self, cost, pairwise_ious, num_gt, valid_mask):
    allowed = torch.isfinite(cost) & (cost < 1e7)
    match = torch.zeros_like(cost,dtype=torch.bool)
    for j in range(num_gt):
        ids = torch.where(allowed[:,j])[0]
        if not len(ids): continue
        k0 = min(self.topk,len(ids))
        k = max(1,int(torch.topk(pairwise_ious[ids,j],k0).values.sum()))
        chosen = ids[torch.topk(cost[ids,j],min(k,len(ids)),largest=False).indices]
        match[chosen,j] = True
    conflict = match.sum(1) > 1
    if conflict.any():
        contested = cost[conflict].masked_fill(~match[conflict],float('inf'))
        chosen = contested.argmin(1)
        match[conflict] = False
        match[conflict,chosen] = True
    fg = match.any(1)
    idx = match.long().argmax(1)
    assigned = torch.where(fg,idx+1,torch.zeros_like(idx))
    return pairwise_ious[fg,idx[fg]],assigned


def install_assignment_fixes(topk):
    from utils.assigner import DynamicSoftLabelAssigner
    from utils.loss import ComputeLoss
    if not getattr(DynamicSoftLabelAssigner,'_revision_installed',False):
        original = DynamicSoftLabelAssigner.assign
        @torch.no_grad()
        def assign(self,scores,boxes,priors,gt,labels):
            self.topk = topk
            return original(self,scores.float(),boxes.float(),priors.float(),gt.float(),labels)
        DynamicSoftLabelAssigner.assign = assign
        DynamicSoftLabelAssigner.dynamic_k_matching = safe_dynamic_k
        DynamicSoftLabelAssigner._revision_installed = True
        ComputeLoss.build_targets_asa = torch.no_grad()(ComputeLoss.build_targets_asa)


@torch.no_grad()
def copy_teacher_buffers(teacher,student):
    source = dict(student.named_buffers())
    for name,value in teacher.named_buffers(): value.copy_(source[name])


class RunAudit:
    def __init__(self,directory,opt,device):
        self.directory,self.opt,self.device = Path(directory),opt,device
        self.started = time.perf_counter()
        self.times,self.epochs,self.steps = [],[],0
        self.peak_allocated,self.peak_reserved = 0,0

    def begin_epoch(self):
        if self.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(self.device)

    def begin(self):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        self.tick = time.perf_counter()

    def end(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            self.peak_allocated = max(self.peak_allocated,torch.cuda.max_memory_allocated(self.device))
            self.peak_reserved = max(self.peak_reserved,torch.cuda.max_memory_reserved(self.device))
        self.times.append(time.perf_counter()-self.tick)

    def record_epoch(self,epoch,results):
        self.epochs.append({'epoch':epoch+1,'metrics':list(map(float,results))})
        best = max(self.epochs,key=lambda row:(.1*row['metrics'][2]+.1*row['metrics'][3]+.8*row['metrics'][4],row['epoch']))
        output = {'variant':self.opt.variant,'seed':self.opt.seed,'best':best,'epochs':self.epochs,
                  'selection_rule':'0.1 AP50 + 0.1 AP75 + 0.8 AP50:95; student EMA',
                  'options':vars(self.opt)}
        (self.directory/'revision_metrics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
        steady = self.times[100:]
        profile = {'wall_seconds_since_training_start':time.perf_counter()-self.started,
                   'training_iteration_seconds_total':sum(self.times),'iterations':len(self.times),
                   'optimizer_steps':self.steps,'timing_warmup_iterations':100,
                   'steady_iteration_ms_mean':float(np.mean(steady)*1000) if steady else None,
                   'steady_iteration_ms_std':float(np.std(steady,ddof=1)*1000) if len(steady)>1 else None,
                   'peak_allocated_GiB':self.peak_allocated/2**30,'peak_reserved_GiB':self.peak_reserved/2**30,
                   'precision':'AMP' if self.device.type=='cuda' else 'FP32',
                   'scope':'Training iteration includes dataloading/branches/backward/update, excludes validation/checkpoint saving; all iterations synchronised.'}
        (self.directory/'training_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')


def save_teacher_snapshot(directory,epoch,epochs,teacher,daat,opt):
    from copy import deepcopy
    milestones = {max(1,math.ceil(epochs*p)) for p in (.1,.25,.5,1)}
    if epoch+1 not in milestones: return
    output = {'epoch':epoch,'model':deepcopy(teacher).half(),'daat':daat.state_dict() if daat else None,
              'options':vars(opt),'purpose':'Offline pseudo-label diagnostics; not a resumable training checkpoint'}
    torch.save(output,Path(directory)/'weights'/('teacher_epoch_%04d.pt'%(epoch+1)))
