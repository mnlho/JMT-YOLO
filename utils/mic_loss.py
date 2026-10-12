import torch
import torch.nn.functional as F

def cosine_similarity_loss(ps1, ps2):
    cos_sim = F.cosine_similarity(ps1, ps2, dim=1)
    loss = 1 - cos_sim
    return loss.mean()



def log_cosh_cosine_similarity_loss(ps1, ps2):
    cos_sim = F.cosine_similarity(ps1, ps2, dim=1)
    loss = torch.log(torch.cosh(1 - cos_sim))  # 使用 log(cosh) 平滑损失
    return loss.mean()





