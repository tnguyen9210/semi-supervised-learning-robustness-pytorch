
import torch
import torch.nn.functional as F


def clamp_by_pnorm(x, order, eps):
    norm = x.abs().pow(order).sum((1,2,3), keepdim=True).pow(1./order)
    factor = torch.min(eps/norm, torch.ones_like(norm))
    return x*factor

def normalize_by_pnorm(x, order, xi=1e-6):
    norm = x.abs().pow(order).sum((1,2,3), keepdim=True).pow(1./order)
    norm = torch.max(norm, torch.ones_like(norm)*xi)
    return x/norm

def kl_div(q_logit, p_logit):
    p_prob = F.softmax(p_logit, dim=1)
    diff = (F.log_softmax(p_logit, dim=1) - F.log_softmax(q_logit, dim=1))
    kl_div = torch.mean(torch.sum(p_prob * diff, dim=1))
    return kl_div


# def normalize(v):
#     v = v / (1e-12 + reduce_max(v.abs(), range(1, len(v.shape))))
#     v = v / (1e-6 + v.pow(2).sum(dim=(1,2,3),keepdim=True)).sqrt()
#     return v

# def reduce_max(v, idx_list):
#     for i in idx_list:
#         v = v.max(dim=i, keepdim=True)[0]
#     return v

    
# def kl_div(logit_yadv, logit_y):
#     logit_yadv = F.log_softmax(logit_yadv, dim=1)
#     logit_y = F.softmax(logit_y, dim=1)
#     return F.kl_div(logit_yadv, logit_y, reduction='batchmean')


# def kl_div(q_logit, p_logit):
#     q = q_logit.softmax(1)
#     qlogp = (q * logsoftmax(p_logit)).sum(1)
#     qlogq = (q * logsoftmax(q_logit)).sum(1)
#     return (qlogq - qlogp).mean()


# def logsoftmax(x):
#     xdev = x - x.max(1, keepdim=True)[0]
#     lsm = xdev - xdev.exp().sum(1, keepdim=True).log()
#     return lsm



