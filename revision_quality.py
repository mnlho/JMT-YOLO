"""Offline teacher pseudo-label diagnostics. Never used by the training loop."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def pairwise_iou(a,b):
    a,b = np.asarray(a).reshape(-1,4),np.asarray(b).reshape(-1,4)
    lo = np.maximum(a[:,None,:2],b[None,:,:2])
    hi = np.minimum(a[:,None,2:],b[None,:,2:])
    inter = np.maximum(hi-lo,0).prod(-1)
    area_a = np.maximum(a[:,2:]-a[:,:2],0).prod(-1)
    area_b = np.maximum(b[:,2:]-b[:,:2],0).prod(-1)
    return inter / np.maximum(area_a[:,None]+area_b[None,:]-inter,1e-12)


def match(pred,gt,threshold=.5,class_aware=True):
    # pred: xyxy, score, class; gt: xyxy, class. Confidence-ordered unique GT matching.
    quality = pairwise_iou(pred[:,:4],gt[:,:4])
    used,set_pairs = set(),[]
    for i in np.argsort(-pred[:,4],kind='stable'):
        allowed = [j for j in range(len(gt)) if j not in used and
                   (not class_aware or pred[i,5]==gt[j,4]) and quality[i,j]>=threshold]
        if allowed:
            j = max(allowed,key=lambda j:quality[i,j]);used.add(j)
            set_pairs.append((int(i),int(j),float(quality[i,j])))
    return set_pairs


def prf(tp,npred,ngt):
    p,r = tp/npred if npred else 0.,tp/ngt if ngt else 0.
    return {'TP':tp,'FP':npred-tp,'FN':ngt-tp,'precision':p,'recall':r,'F1':2*p*r/(p+r) if p+r else 0.}


def summarise(samples,bins=15):
    npred=sum(len(p) for p,g,c in samples);ngt=sum(len(g) for p,g,c in samples)
    tp=sum(len(match(p,g,.5)) for p,g,c in samples)
    tp75=sum(len(match(p,g,.75)) for p,g,c in samples)
    geometry=[];wrong=0;confusion={};conf=[];correct=[];classes=set()
    perclass={}
    for p,g,c in samples:
        classes.update(p[:,5].astype(int).tolist());classes.update(g[:,4].astype(int).tolist())
        pairs=match(p,g,.5,False);geometry.extend(q for i,j,q in pairs)
        for i,j,q in pairs:
            wrong+=int(p[i,5]!=g[j,4]);key='%d->%d'%(g[j,4],p[i,5]);confusion[key]=confusion.get(key,0)+1
        hits=np.zeros(len(c));hits[[i for i,j,q in match(c,g,.5)]]=1
        conf.extend(c[:,4].tolist());correct.extend(hits.tolist())
    for cls in sorted(classes):
        counts=[(p[p[:,5]==cls],g[g[:,4]==cls]) for p,g,c in samples]
        perclass[str(cls)]=prf(sum(len(match(p,g)) for p,g in counts),sum(len(p) for p,g in counts),sum(len(g) for p,g in counts))
    reliability=[];ece=0.;conf,correct=np.asarray(conf),np.asarray(correct)
    for i in range(bins):
        lo,hi=i/bins,(i+1)/bins
        use=(conf>=lo)&((conf<hi) if i<bins-1 else (conf<=hi))
        n=int(use.sum());mean=float(conf[use].mean()) if n else None;acc=float(correct[use].mean()) if n else None
        if n:ece+=n/max(1,len(conf))*abs(mean-acc)
        reliability.append({'bin_low':lo,'bin_high':hi,'count':n,'mean_confidence':mean,'accuracy':acc})
    output={'images':len(samples),'IoU_0.5':prf(tp,npred,ngt),'IoU_0.75':prf(tp75,npred,ngt),
            'pseudo_labels_per_image':npred/max(1,len(samples)),
            'zero_pseudo_label_image_fraction':sum(len(p)==0 for p,g,c in samples)/max(1,len(samples)),
            'class_error_given_class_agnostic_IoU_match':wrong/len(geometry) if geometry else None,
            'class_agnostic_matched_mean_IoU':float(np.mean(geometry)) if geometry else None,
            'class_agnostic_matched_median_IoU':float(np.median(geometry)) if geometry else None,
            'class_agnostic_matching_coverage':len(geometry)/ngt if ngt else None,
            'class_confusion_gt_to_prediction':confusion,'per_class':perclass,
            'calibration_ECE':ece if len(conf) else None,
            'calibration_Brier':float(((conf-correct)**2).mean()) if len(conf) else None,
            'calibration_candidates':len(conf),'reliability':reliability,
            'definitions':'Unique GT, confidence-ordered matching. Classification noise uses class-agnostic IoU>=0.5 matches. Calibration correctness uses same-class IoU>=0.5. Undefined conditional means are null; empty-set P/R/F1 are 0.'}
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True);p.add_argument('--images',required=True)
    p.add_argument('--labels',required=True,help='Original target GT YOLO label directory, offline only')
    p.add_argument('--output',required=True);p.add_argument('--device',default='0');p.add_argument('--img-size',type=int,default=960)
    p.add_argument('--half',action='store_true');p.add_argument('--threshold',type=float)
    p.add_argument('--classes',nargs='+',type=int);p.add_argument('--nms-iou',type=float,default=.3);p.add_argument('--max-det',type=int,default=20)
    a=p.parse_args()
    import torch,cv2
    from models.experimental import attempt_load
    from utils.general import non_max_suppression,scale_coords
    from utils.datasets import letterbox
    from utils.torch_utils import select_device
    from revision_support import image_map
    device=select_device(a.device)
    if a.half and device.type=='cpu':raise ValueError('FP16 requires CUDA')
    checkpoint=torch.load(a.weights,map_location='cpu')
    state=checkpoint.get('daat');options=checkpoint.get('options',{})
    threshold=a.threshold if a.threshold is not None else (state['current_threshold'] if state else options.get('conf_thres',.5))
    model=attempt_load(a.weights,map_location=device).eval()
    model=model.half() if a.half else model.float()
    images=image_map(a.images)
    labels={x.stem:x for x in Path(a.labels).rglob('*.txt')}
    if not images.keys()<=labels.keys():raise ValueError('GT missing for image IDs: '+str(sorted(images.keys()-labels.keys())[:5]))
    samples=[]
    with torch.no_grad():
        for key,path in images.items():
            im=cv2.imread(str(path))
            if im is None:raise ValueError('Unreadable '+str(path))
            h,w=im.shape[:2]
            inp=letterbox(im,a.img_size,auto=False)[0]
            x=torch.from_numpy(np.ascontiguousarray(inp[:,:,::-1].transpose(2,0,1))).unsqueeze(0).to(device)
            x=(x.half() if a.half else x.float())/255
            raw=model(x)[0][0]
            selected=non_max_suppression(raw.clone(),threshold,a.nms_iou,multi_label=True,max_det=a.max_det,classes=a.classes)[0]
            candidates=non_max_suppression(raw.clone(),.001,a.nms_iou,multi_label=True,max_det=300,classes=a.classes)[0]
            for out in [selected,candidates]:
                if len(out):scale_coords(x.shape[2:],out[:,:4],im.shape)
            rows=[line.split() for line in labels[key].read_text().splitlines() if line.strip()]
            lb=np.asarray(rows,dtype=float).reshape(-1,5)
            if not np.isfinite(lb).all() or (lb[:,1:]<0).any() or (lb[:,1:]>1).any():raise ValueError('Invalid GT '+str(labels[key]))
            gt=np.zeros((len(lb),5));gt[:,4]=lb[:,0]
            gt[:,0]=(lb[:,1]-lb[:,3]/2)*w;gt[:,2]=(lb[:,1]+lb[:,3]/2)*w
            gt[:,1]=(lb[:,2]-lb[:,4]/2)*h;gt[:,3]=(lb[:,2]+lb[:,4]/2)*h
            if a.classes is not None:gt=gt[np.isin(gt[:,4],a.classes)]
            samples.append((selected.cpu().numpy(),gt,candidates.cpu().numpy()))
    result=summarise(samples)
    result['protocol']={'weights':a.weights,'images':a.images,'labels':a.labels,'threshold':float(threshold),
                        'threshold_mode':'Frozen saved teacher threshold; evaluation does not update DAAT',
                        'candidate_confidence_floor':.001,'candidate_max_det':300,'selected_max_det':a.max_det,
                        'nms_iou':a.nms_iou,'classes':a.classes,'input_size':a.img_size,'half':a.half,
                        'geometry_requirement':'Images and GT must share original geometry. Use target_fake images and matching target_real labels. This measures fixed unaugmented diagnostic inputs, not replayed training batches.'}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with output.with_suffix('.reliability.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['reliability'][0]));writer.writeheader();writer.writerows(result['reliability'])
    print(json.dumps({k:result[k] for k in ['IoU_0.5','IoU_0.75','calibration_ECE','calibration_Brier']},indent=2));print(output)


if __name__=='__main__':main()
