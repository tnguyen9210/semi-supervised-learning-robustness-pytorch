
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixMatch(nn.Module):
    def __init__(self, num_augments, temperature, alpha):
        super(MixMatch, self).__init__()
        self.K = num_augments
        self.T = temperature
        self.alpha = alpha
        # self.beta_dist = torch.distributions.beta.Beta(alpha, alpha)

    def sharpen(self, prob):
        prob = prob.pow(1/self.T)
        return prob/prob.sum(1, keepdim=True)

    def forward(self, net, unlbl_x1, unlbl_x2, lbl_x, lbl_y):
        batch_size = unlbl_x1.shape[0]
        
        # compute guessed label (prob) for K unlbl augments
        with torch.no_grad():
            unlbl_logit1 = net(unlbl_x1)
            unlbl_logit2 = net(unlbl_x2)

            p = (torch.softmax(unlbl_logit1, dim=1) \
                 + torch.softmax(unlbl_logit2, dim=1)) / 2
            pt = p**(1/self.T)
            unlbl_prob = pt/torch.sum(pt, dim=1, keepdim=True)
            unlbl_prob = unlbl_prob.detach()

        # mixup
        all_x = torch.cat([lbl_x, unlbl_x1, unlbl_x2], dim=0)
        all_y = torch.cat([lbl_y, unlbl_prob, unlbl_prob], dim=0)

        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1-lam)

        idxes = torch.randperm(all_x.shape[0])

        all_xa = all_x
        all_xb = all_x[idxes]
        all_ya = all_y
        all_yb = all_y[idxes]

        mixed_x = lam*all_xa + (1-lam)*all_xb
        mixed_y = lam*all_ya + (1-lam)*all_yb

        # interleave labeled and unlabeled batches
        # to get correct batchnorm calculation
        mixed_x = list(torch.split(mixed_x, batch_size))
        mixed_x = interleave(mixed_x, batch_size)
        # net.update_batch_stats(False)

        mixed_logit = []
        for x in mixed_x:
            mixed_logit.append(net(x))

        # put interleaved samples back
        mixed_logit = interleave(mixed_logit, batch_size)
        # net.update_batch_stats(True)
        lbl_logit = mixed_logit[0]
        unlbl_logit = torch.cat(mixed_logit[1:], dim=0)
        
        return unlbl_logit, mixed_y[batch_size:], lbl_logit, mixed_y[:batch_size]
        

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


