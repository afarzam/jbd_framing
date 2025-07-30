import torch.nn.functional as F, torch

def l2_recon(v, v_hat):
    return F.mse_loss(v_hat, v)

def orth_penalty(vg, vf):
    # minimise squared cosine similarity
    dot = (vg * vf).sum(dim=-1)
    return (dot ** 2).mean()

def info_nce(anchor, pos, neg, T=0.1):
    """
    anchor, pos : (B, d)
    neg         : (B*K, d)      # negatives (all other goals)
    """
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(pos,    dim=-1)
    n = F.normalize(neg,    dim=-1)

    pos_sim = (a * p).sum(-1, keepdim=True) / T          # (B,1)
    neg_sim = a @ n.t() / T                              # (B, B*K)

    logits  = torch.cat([pos_sim, neg_sim], dim=1)
    labels  = torch.zeros(len(a), dtype=torch.long,
                          device=a.device)
    return F.cross_entropy(logits, labels)
