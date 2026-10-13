import torch
import torch.nn.functional as F

# ---------- Outer objectives ----------

def ce_outer(logits, y):
    return F.cross_entropy(logits, y)

def trades_outer(logits_clean, logits_adv, y, beta: float):
    ce = F.cross_entropy(logits_clean, y)
    p_clean = F.log_softmax(logits_clean.detach(), dim=1).exp()
    kl = F.kl_div(F.log_softmax(logits_adv, dim=1), p_clean, reduction="batchmean")
    return ce + beta * kl

# ---------- Inner losses (to MAXIMIZE) ----------

def ce_inner(logits, y):
    return F.cross_entropy(logits, y, reduction="none")

# def dlr_inner(logits, y, targeted: bool = False, y_target=None):
#     z = logits
#     if not targeted:
#         z_y = z.gather(1, y[:, None]).squeeze(1)
#         z_sorted, _ = z.sort(dim=1, descending=True)
#         num = z_y - z_sorted[:, 1]
#         den = (z_sorted[:, 0] - z_sorted[:, -1]).clamp_min(1e-12)
#         return -(num / den)
#     else:
#         assert y_target is not None
#         z_t = z.gather(1, y_target[:, None]).squeeze(1)
#         z_sorted, _ = z.sort(dim=1, descending=True)
#         num = z_sorted[:, 0] - z_t
#         den = (z_sorted[:, 0] - z_sorted[:, -1]).clamp_min(1e-12)
#         return (num / den)

# def margin_inner(logits, y):
#     z = logits
#     z_y = z.gather(1, y[:, None]).squeeze(1)
#     z_others = z.clone()
#     z_others.scatter_(1, y[:, None], float("-inf"))
#     m = z_y - z_others.max(dim=1).values
#     return -m

# def cw_inner(logits, y, kappa: float = 0.0):
#     z = logits
#     z_y = z.gather(1, y[:, None]).squeeze(1)
#     z_others = z.clone()
#     z_others.scatter_(1, y[:, None], float("-inf"))
#     m = z_y - z_others.max(dim=1).values
#     return torch.clamp(m + kappa, min=0)

# TRADES inner is KL(p(x) || p(x_adv)) computed in trainers

INNER_LOSSES = {
    "ce": ce_inner,
    # "dlr": dlr_inner,
    # "margin": margin_inner,
    # "cw": cw_inner,
}
