
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks_delta import *
from core.utils import *


class ATLoss(nn.Module):
    def __init__(self):
        super(ATLoss, self).__init__()
        linf_eps = 0.3
        linf_eps_iter = 0.05
        linf_niters = 20
        l2_eps = 3.
        l2_eps_iter = 0.5
        l2_niters = 20
    
        self.loss_fn = F.cross_entropy
        
        # self.attacker = GradientSignAttack(loss_fn=self.loss_fn, eps=linf_eps)
        self.attacker = LinfPGDAttack(
            loss_fn=self.loss_fn, num_iters=linf_niters, eps=linf_eps, eps_iter=linf_eps_iter)
        
        # self.attacker = GradientAttack(loss_fn=self.loss_fn, eps=l2_eps)
        # self.attacker = L2PGDAttack(
        #     loss_fn=self.loss_fn, num_iters=l2_niters, eps=l2_eps, eps_iter=l2_eps_iter)

    def forward(self, net, x, y=None): 
        net.update_batch_stats(False)
        x_adv, _ = self.attacker.perturb(net, x, y)
        net.update_batch_stats(True)
        logit_yadv = net(x_adv)
        loss = self.loss_fn(logit_yadv, y)
        return loss 
    
    
