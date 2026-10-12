import torch
import torch.nn.functional as F
from utils.general import bbox_iou

class DynamicSoftLabelAssigner:
    """
    Computes matching between predictions and ground truth with dynamic soft label assignment.
    """
    def __init__(self, topk=13, iou_factor=2.0, dist_factor=0.1):
        self.topk = topk
        self.iou_factor = iou_factor
        self.dist_factor = dist_factor

    def assign(self, pred_scores, decoded_bboxes, priors, gt_bboxes, gt_labels):
        """
        Args:
            pred_scores (Tensor): [num_priors, num_classes]
            decoded_bboxes (Tensor): [num_priors, 4] in [x1, y1, x2, y2] format
            priors (Tensor): [num_priors, 4] in [cx, cy, stride_w, stride_h] format (YOLO anchors)
            gt_bboxes (Tensor): [num_gts, 4] in [x1, y1, x2, y2] format
            gt_labels (Tensor): [num_gts]
            teacher_scores (Tensor, optional): [num_priors, num_classes] - Teacher predictions for consistency
        
        Returns:
            assigned_gt_inds (Tensor): [num_priors] - 0: ignore/bg, >0: gt index (1-based)
            assigned_labels (Tensor): [num_priors]
            pos_mask (Tensor): [num_priors] - boolean mask for positive samples
            neg_mask (Tensor): [num_priors] - boolean mask for HARD negative samples (candidates that failed matching)
        """
        INF = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)

        # assigned_gt_inds: 0 means background, >0 means gt_index + 1
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        
        # Masks for Triplet Loss
        pos_mask = torch.zeros(num_bboxes, dtype=torch.bool, device=decoded_bboxes.device)
        neg_mask = torch.zeros(num_bboxes, dtype=torch.bool, device=decoded_bboxes.device)

        if num_gt == 0 or num_bboxes == 0:
            return assigned_gt_inds, assigned_labels, pos_mask, neg_mask

        # 1. Spatial Prior / Pre-check: Check if anchor center is inside GT box
        # priors: [cx, cy, stride_w, stride_h]
        # Optimized to avoid large tensor allocation [num_priors, num_gt, 4]
        
        # Use broadcasting with boolean masks (much smaller memory footprint)
        # priors[:, 0] is cx, priors[:, 1] is cy
        # gt_bboxes: [x1, y1, x2, y2]
        
        # Check x: x1 < cx < x2
        is_in_gts_x = (priors[:, 0].unsqueeze(1) > gt_bboxes[:, 0].unsqueeze(0)) & \
                      (priors[:, 0].unsqueeze(1) < gt_bboxes[:, 2].unsqueeze(0))
                      
        # Check y: y1 < cy < y2
        is_in_gts_y = (priors[:, 1].unsqueeze(1) > gt_bboxes[:, 1].unsqueeze(0)) & \
                      (priors[:, 1].unsqueeze(1) < gt_bboxes[:, 3].unsqueeze(0))
                      
        is_in_gts = is_in_gts_x & is_in_gts_y # [num_priors, num_gt] bool
        
        valid_mask = is_in_gts.any(dim=1) # [num_priors] - These are CANDIDATES

        
        # Record candidates as potential negatives (we will remove positives later)
        neg_mask[valid_mask] = True

        if not valid_mask.any():
            return assigned_gt_inds, assigned_labels, pos_mask, neg_mask

        # Filter valid bboxes and scores
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        valid_prior = priors[valid_mask]
        
        # 2. Compute Cost Matrix
        # IoU Cost
        # ious: [num_valid, num_gt]
        # We need to broadcast valid_decoded_bbox to [4, num_valid, 1] to compute pairwise IoU with gt_bboxes [num_gt, 4]
        # bbox_iou expects box1 as [4, N, 1] and box2 as [M, 4] (which it transposes to [4, M])
        ious = bbox_iou(valid_decoded_bbox.T.unsqueeze(2), gt_bboxes, x1y1x2y2=True, CIoU=True).clamp(min=0)
        iou_cost = -torch.log(ious + 1e-8)
        
        # Classification Cost
        # gt_labels: [num_gt]
        # valid_pred_scores: [num_valid, num_classes]
        # We want the score of the target class for each GT
        # gt_scores: [num_valid, num_gt]
        gt_scores = valid_pred_scores[:, gt_labels]
        cls_cost = -torch.log(gt_scores + 1e-8)

        # Distance Cost (Optional, but good for stability)
        gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:4]) / 2.0
        # valid_prior[:, 2] is stride (approximate)
        strides = valid_prior[:, 2] 
        
        # Optimized: Use torch.cdist for efficient pairwise distance calculation
        # dist_cost: [num_valid, num_gt]
        # Ensure inputs are the same type (Float) to avoid RuntimeError: expected scalar type Float but found Double
        dist_cost = torch.cdist(valid_prior[:, :2].float(), gt_center.float(), p=2)
        dist_cost.div_(strides[:, None]) # In-place division
        
        # Total Cost
        # cost_matrix: [num_valid, num_gt]
        # Optimized: Use in-place operations to reduce memory peak
        cost_matrix = cls_cost
        cost_matrix.add_(iou_cost, alpha=self.iou_factor)
        cost_matrix.add_(dist_cost, alpha=self.dist_factor)
        
        # Clean up large intermediate tensors immediately
        del iou_cost, dist_cost, gt_scores
        
        # Mask out invalid pairs (not in spatial prior)
        # is_in_gts_valid: [num_valid, num_gt]
        is_in_gts_valid = is_in_gts[valid_mask]
        cost_matrix[~is_in_gts_valid] = INF

        # 3. Dynamic K Matching
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, ious, num_gt, valid_mask
        )
        
        # 4. Update Results
        # matched_gt_inds has shape [num_valid]
        # values > 0 indicate match to gt_index+1
        
        # Map back to full size
        assigned_gt_inds[valid_mask] = matched_gt_inds
        
        # Update masks
        # Positives: assigned_gt_inds > 0
        pos_mask = assigned_gt_inds > 0
        
        # Negatives: valid_mask is True BUT assigned_gt_inds is 0
        # These are the "candidates that failed matching" -> Hard Negatives
        neg_mask = valid_mask & (~pos_mask)
        
        if num_gt > 0 and pos_mask.any():
            # Assign labels
            # assigned_gt_inds[pos_mask] is 1-based index
            pos_assigned_gt_inds = assigned_gt_inds[pos_mask] - 1
            assigned_labels[pos_mask] = gt_labels[pos_assigned_gt_inds]
            
        return assigned_gt_inds, assigned_labels, pos_mask, neg_mask

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        """
        cost: [num_valid, num_gt]
        pairwise_ious: [num_valid, num_gt]
        """
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        
        # Select top-k IoUs for each GT to estimate dynamic K
        # topk_ious: [topk, num_gt]
        n_candidate_k = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, n_candidate_k, dim=0)
        
        # dynamic_ks: [num_gt]
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        
        # Assign for each GT
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[pos_idx, gt_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        # Filter conflicts: if an anchor is matched to multiple GTs, pick the one with min cost
        anchor_matching_gt = matching_matrix.sum(1) # [num_valid]
        
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] = 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin] = 1
            
        # Prepare result
        # matched_gt_inds: [num_valid] (0 for bg, 1-based for gt)
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        matched_gt_inds = matching_matrix.argmax(1) + 1
        matched_gt_inds[~fg_mask_inboxes] = 0
        
        return pairwise_ious[fg_mask_inboxes, matching_matrix.argmax(1)[fg_mask_inboxes]], matched_gt_inds
