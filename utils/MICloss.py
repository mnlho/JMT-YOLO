import torch
import torch.nn.functional as F
import torch.nn as nn

class KLDivergenceLossForOutputs(nn.Module):
    def __init__(self):
        super(KLDivergenceLossForOutputs, self).__init__()

    def forward(self, student_output, teacher_output):
        """
        Computes the KL divergence loss between student and teacher model outputs.

        Args:
            student_output: Tensor of shape (batch_size, num_classes, H, W) - student model logits
            teacher_output: Tensor of shape (batch_size, num_classes, H, W) - teacher model logits

        Returns:
            loss: KL divergence loss value
        """

        # Convert student output to log-probabilities
        student_log_probs = F.log_softmax(student_output, dim=1)
        # Convert teacher output to probabilities
        teacher_probs = F.softmax(teacher_output, dim=1)

        # Compute KL divergence loss
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return loss

# Example usage:
# student_output = student_model(pred_input) (shape: [batch_size, num_classes, H, W])
# teacher_output = teacher_model(pred_input) (shape: [batch_size, num_classes, H, W])
# mic_loss_fn = KLDivergenceLossForOutputs()
# mic_loss = mic_loss_fn(student_output, teacher_output)
