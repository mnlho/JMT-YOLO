# Loss functions
# 添加ASA后的loss计算

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general import bbox_iou
from utils.mic_loss import cosine_similarity_loss, log_cosh_cosine_similarity_loss
# from utils.mic_loss import cosine_similarity_loss
from utils.torch_utils import is_parallel
from utils.assigner import DynamicSoftLabelAssigner


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def dice_loss_multi_class(probs_teacher, probs_student, smooth=1e-8):
    # 初始化 Dice 损失
    dice_loss_total = 0.0

    # 遍历每个类别
    for class_idx in range(probs_teacher.shape[1]):
        # 提取当前类别的概率
        probs_teacher_class = probs_teacher[:, class_idx, :, :].contiguous().view(-1)
        probs_student_class = probs_student[:, class_idx, :, :].contiguous().view(-1)

        # 计算 Dice 系数
        intersection = torch.sum(probs_teacher_class * probs_student_class)
        union = torch.sum(probs_teacher_class) + torch.sum(probs_student_class)
        dice = (2.0 * intersection + smooth) / (union + smooth)

        # 累加 Dice 损失
        dice_loss_total += 1.0 - dice

    # 返回平均 Dice 损失
    return dice_loss_total / probs_teacher.shape[1]


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # ASA Assigner
        iou_factor = h.get('iou_factor', 3.0)
        dist_factor = h.get('dist_factor', 0.1)
        self.asa_assigner = DynamicSoftLabelAssigner(topk=13, iou_factor=iou_factor, dist_factor=dist_factor)

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, mask_only=False, is_target_domain=False, use_asa=False):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        triplet_negs = []
        if is_target_domain or use_asa:
             tcls, tbox, indices, anchors, triplet_negs = self.build_targets_asa(p, targets, None)
        else:
             tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # if mask_only:
        #     # Initialize mask loss
        #     lcontrastive = torch.zeros(1, device=device)
        #     for i, pi in enumerate(p):
        #         b, a, gj, gi = indices[i]
        #         tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
        #         if b.shape[0] > 0:
        #             ps = pi[b, a, gj, gi]
        #             pxy = ps[:, :2].sigmoid() * 2. - 0.5
        #             pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
        #             pbox = torch.cat((pxy, pwh), 1)  # predicted box
        #             iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
        #             # lbox = lbox + (1.0 - iou).mean()  # iou loss
        #             score_iou = iou.detach().clamp(0).type(tobj.dtype)
        #
        #             tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio
        #
        #             # if self.nc > 1:  # cls loss (only if multiple classes)
        #             #     t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
        #             #     t[range(b.shape[0]), tcls[i]] = self.cp
        #             #     lcls = lcls + self.BCEcls(ps[:, 5:], t)  # BCE
        #         obji = cosine_similarity_loss(pi[..., 4], tobj)
        #         lcontrastive = lcontrastive + obji
        #         # lcontrastive = lcontrastive / (lcontrastive.norm(p=2) + 1e-6)  # L2归一化
        #     # bs = tobj.shape[0]  # batch size
        #     return lcontrastive, torch.tensor([0, 0, 0, lcontrastive.item()]).detach()

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox = lbox + (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls = lcls + self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj = lobj + obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox = lbox * self.hyp['box']
        lobj = lobj * self.hyp['obj']
        lcls = lcls * self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        # Triplet Loss Calculation
        ltriplet = torch.zeros(1, device=device)
        # if is_target_domain and len(triplet_negs) > 0:
        #     if len(triplet_negs) > 0:
        #         all_negs = torch.cat(triplet_negs, dim=1).T # [N_negs, 5]

        #         max_negs = 1000
        #         if all_negs.shape[0] > max_negs:
        #             perm = torch.randperm(all_negs.shape[0])[:max_negs]
        #             all_negs = all_negs[perm]

                # For each negative, we need a corresponding positive/anchor
                # But wait, Triplet Loss usually requires (Anchor, Positive, Negative) triplets.
                # Here:
                # Anchor = Teacher Feature (or Pseudo Label Embedding)
                # Positive = Student Feature at Positive Location
                # Negative = Student Feature at Negative Location
                
                # Since we don't have Teacher Features passed in here easily without changing the signature too much,
                # and we are inside ComputeLoss which only sees Student Predictions 'p',
                # We can implement a simplified "Mining" loss:
                # Minimize confidence of Hard Negatives
                
                # Extract Negative Scores
                # all_negs: [b, l, a, gj, gi]


                # neg_scores = []
                # for row in all_negs:
                #     b, l, a, gj, gi = row.long()
                    # p[l] is [bs, 3, h, w, 85]
                    # score is objectness * class_prob
                    # or just objectness
                    # neg_scores.append(p[l][b, a, gj, gi, 4]) # Objectness logits
                
                # if len(neg_scores) > 0:
                #     neg_scores = torch.stack(neg_scores)
                    # We want these scores to be low (background)
                    # Target is 0
                    # ltriplet = self.BCEobj(neg_scores, torch.zeros_like(neg_scores)) * 0.01 # Weight it down
        
        loss = lbox + lobj + lcls + ltriplet
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, int(gain[3]) - 1), gi.clamp_(0, int(gain[2]) - 1)))  # integer grid bounds, compatible with newer torch
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def build_targets_asa(self, p, targets, imgs):
        # p: list of predictions [P3, P4, P5]
        # targets: [num_targets, 6] (image_idx, class, x, y, w, h) - Pseudo Labels (Normalized)
        # imgs: input images [bs, 3, h, w] (used for shape)
        
        # Initialize lists for each layer
        layer_indices = [[] for _ in range(self.nl)]
        layer_tbox = [[] for _ in range(self.nl)]
        layer_anch = [[] for _ in range(self.nl)]
        layer_tcls = [[] for _ in range(self.nl)]
        layer_triplet_negs = [[] for _ in range(self.nl)]
        
        device = targets.device
        bs = p[0].shape[0]
        
        # Iterate over each image in the batch
        for b_idx in range(bs):
            # Get GTs for current image
            t = targets[targets[:, 0] == b_idx]
            if t.shape[0] == 0:
                continue
            
            # Prepare GTs in absolute coordinates (for ASA)
            # targets are normalized [x, y, w, h]
            # We need to scale them to the input image size
            # Assuming imgs is available, or we can infer from p[0] * stride[0]
            # p[0] stride is 8. p[0].shape[-2:] * 8 = img_size
            img_h = p[0].shape[2] * self.stride[0]
            img_w = p[0].shape[3] * self.stride[0]
            
            gt_bboxes_norm = t[:, 2:6] # xywh normalized
            gt_bboxes_abs = gt_bboxes_norm.clone()
            gt_bboxes_abs[:, 0] *= img_w
            gt_bboxes_abs[:, 2] *= img_w
            gt_bboxes_abs[:, 1] *= img_h
            gt_bboxes_abs[:, 3] *= img_h
            
            # xywh -> xyxy
            gt_bboxes_xyxy = torch.zeros_like(gt_bboxes_abs)
            gt_bboxes_xyxy[:, 0] = gt_bboxes_abs[:, 0] - gt_bboxes_abs[:, 2] / 2
            gt_bboxes_xyxy[:, 1] = gt_bboxes_abs[:, 1] - gt_bboxes_abs[:, 3] / 2
            gt_bboxes_xyxy[:, 2] = gt_bboxes_abs[:, 0] + gt_bboxes_abs[:, 2] / 2
            gt_bboxes_xyxy[:, 3] = gt_bboxes_abs[:, 1] + gt_bboxes_abs[:, 3] / 2
            
            gt_labels = t[:, 1].long()
            
            # Iterate over each feature level (P3, P4, P5)
            for i in range(self.nl): 
                anchors = self.anchors[i] # [3, 2]
                stride = int(self.stride[i])
                
                # Prediction for current image and layer: [3, h, w, 85]
                pi = p[i][b_idx] 
                na, h, w, _ = pi.shape
                
                # Generate Grid
                grid_y, grid_x = torch.meshgrid([torch.arange(h, device=device), torch.arange(w, device=device)])
                grid_xy = torch.stack((grid_x, grid_y), 2).float() # [h, w, 2]
                
                # Decode Predictions to Absolute Coordinates
                # pred_xy = (sigmoid(tx) * 2 - 0.5 + grid_x) * stride
                pred_xy = (pi[..., :2].sigmoid() * 2. - 0.5 + grid_xy) * stride
                # pred_wh = (sigmoid(tw) * 2) ** 2 * anchor_wh * stride
                pred_wh = (pi[..., 2:4].sigmoid() * 2) ** 2 * anchors.view(na, 1, 1, 2) * stride
                
                # [3, h, w, 4] -> [N, 4]
                decoded_bboxes = torch.cat((pred_xy - pred_wh/2, pred_xy + pred_wh/2), dim=-1).view(-1, 4)
                
                # Construct Priors (Anchors) for Distance Cost
                # [3, h, w, 4] -> [N, 4] (cx, cy, stride, stride)
                priors = torch.zeros_like(decoded_bboxes)
                priors[:, :2] = pred_xy.view(-1, 2) # Use predicted center or grid center? ASA uses prior center.
                # Using grid center as prior center is more stable for "Anchor-Free" style, 
                # but YOLO is Anchor-Based. Let's use the grid center + anchor shift as prior.
                # For YOLO, the "Anchor" is the grid point + anchor box.
                # Let's use the grid center * stride as the prior center for simplicity and stability.
                prior_center_xy = (grid_xy * stride).unsqueeze(0).repeat(na, 1, 1, 1).view(-1, 2)
                priors[:, :2] = prior_center_xy
                priors[:, 2:] = stride
                
                # Classification Scores
                pred_scores = pi[..., 5:].sigmoid().view(-1, self.nc)
                
                # --- Run ASA ---
                assigned_gt_inds, assigned_labels, pos_mask, neg_mask = self.asa_assigner.assign(
                    pred_scores, decoded_bboxes, priors, gt_bboxes_xyxy, gt_labels
                )
                
                # --- Process Matches ---
                if pos_mask.any():
                    # Get indices of positive samples
                    pos_idxs = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                    
                    # Map back to (anchor_idx, grid_y, grid_x)
                    a = torch.div(pos_idxs, (h * w), rounding_mode='floor')
                    rem = pos_idxs % (h * w)
                    gj = torch.div(rem, w, rounding_mode='floor')
                    gi = rem % w
                    
                    # Append Indices: (image_idx, anchor_idx, grid_y, grid_x)
                    layer_indices[i].append((torch.full_like(a, b_idx), a, gj, gi))
                    
                    # Prepare Targets for Loss
                    # We need to convert matched GTs back to YOLO format relative to the grid cell
                    matched_gt_idx = assigned_gt_inds[pos_mask] - 1
                    matched_gt_norm = t[matched_gt_idx] # Normalized xywh
                    
                    # Normalized -> Grid Coordinates
                    gxy = matched_gt_norm[:, 2:4]
                    gxy[:, 0] *= w # x * width
                    gxy[:, 1] *= h # y * height
                    
                    gwh = matched_gt_norm[:, 4:6]
                    gwh[:, 0] *= w
                    gwh[:, 1] *= h
                    
                    # tbox: [gxy - gij, gwh]
                    gij = torch.stack((gi, gj), 1).float()
                    layer_tbox[i].append(torch.cat((gxy - gij, gwh), 1))
                    
                    layer_anch[i].append(anchors[a])
                    layer_tcls[i].append(matched_gt_norm[:, 1].long())
                
                # --- Process Triplet Negatives ---
                if neg_mask.any():
                    neg_idxs = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)
                    a = torch.div(neg_idxs, (h * w), rounding_mode='floor')
                    rem = neg_idxs % (h * w)
                    gj = torch.div(rem, w, rounding_mode='floor')
                    gi = rem % w
                    # Store (batch_idx, layer_idx, anchor_idx, grid_y, grid_x)
                    # We need layer_idx to know which feature map to pull from
                    layer_triplet_negs[i].append((torch.full_like(a, b_idx), torch.full_like(a, i), a, gj, gi))

        # Aggregate results per layer
        indices, tbox, anch, tcls = [], [], [], []
        triplet_negs = []
        
        for i in range(self.nl):
            if len(layer_indices[i]) > 0:
                # Concatenate indices
                b_list, a_list, gj_list, gi_list = zip(*layer_indices[i])
                b = torch.cat(b_list, 0)
                a = torch.cat(a_list, 0)
                gj = torch.cat(gj_list, 0)
                gi = torch.cat(gi_list, 0)
                indices.append((b, a, gj, gi))
                
                tbox.append(torch.cat(layer_tbox[i], 0))
                anch.append(torch.cat(layer_anch[i], 0))
                tcls.append(torch.cat(layer_tcls[i], 0))
                
                if len(layer_triplet_negs[i]) > 0:
                    for item in layer_triplet_negs[i]:
                        triplet_negs.append(torch.stack(item))
            else:
                # No targets for this layer across the whole batch
                indices.append((torch.tensor([], dtype=torch.long, device=device), 
                                torch.tensor([], dtype=torch.long, device=device), 
                                torch.tensor([], dtype=torch.long, device=device), 
                                torch.tensor([], dtype=torch.long, device=device)))
                tbox.append(torch.tensor([], device=device))
                anch.append(torch.tensor([], device=device))
                tcls.append(torch.tensor([], device=device))

        return tcls, tbox, indices, anch, triplet_negs


