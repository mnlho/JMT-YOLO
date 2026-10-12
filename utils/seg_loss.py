import torch
import torch.nn as nn

class CrossEntropySegmentationLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropySegmentationLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes, H, W) - raw model outputs (not softmax-ed)
            targets: Tensor of shape (batch_size, H, W) - ground truth class indices

        Returns:
            loss: Computed cross-entropy loss
        """
        loss = self.loss_fn(logits, targets)
        return loss

# Example usage:
# logits = model_output (assumed shape: [batch_size, num_classes, H, W])
# targets = ground_truth_labels (assumed shape: [batch_size, H, W])
# segloss_fn = CrossEntropySegmentationLoss()
# loss = segloss_fn(logits, targets)


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=None, smooth=1e-6):
        """
        Dice Loss for multi-class segmentation.
        Args:
            ignore_index (int or None): Class index to ignore in the loss calculation.
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Compute Dice Loss.
        Args:
            preds (torch.Tensor): Predicted probabilities, shape [B, C, H, W].
            targets (torch.Tensor): Ground truth labels, shape [B, H, W].
        Returns:
            torch.Tensor: Dice loss value.
        """
        num_classes = preds.size(1)
        preds = torch.softmax(preds, dim=1)  # Convert logits to probabilities

        # One-hot encode targets to match preds shape
        targets_one_hot = torch.zeros_like(preds).scatter_(1, targets.unsqueeze(1), 1)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            mask = mask.unsqueeze(1)  # Shape: [B, 1, H, W]
            preds = preds * mask
            targets_one_hot = targets_one_hot * mask

        # Compute intersection and union
        intersection = torch.sum(preds * targets_one_hot, dim=(2, 3))  # Sum over H, W
        union = torch.sum(preds + targets_one_hot, dim=(2, 3))

        # Compute Dice score per class
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Exclude ignored classes
        if self.ignore_index is not None:
            dice_score[:, self.ignore_index] = 0

        # Average Dice score across all classes
        dice_loss = 1 - torch.mean(dice_score)
        return dice_loss
