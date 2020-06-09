
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks_delta import *
from core.attacks_virtual import *
from core.utils import *


class VATLoss(nn.Module):
    def __init__(self):
        super(VATLoss, self).__init__()
        linf_eps = 8./255.
        linf_eps_iter = 2.0/255.
        linf_niters = 7
        l2_eps = 320./255.
        l2_eps_iter = 80./255.
        l2_niters = 7
        
        self.loss_fn = kl_div

        # self.attacker = GradientSignAttack(loss_fn=self.loss_fn, eps=linf_eps)
        self.attacker = LinfPGDAttack(
            loss_fn=self.loss_fn, num_iters=linf_niters, eps=linf_eps, eps_iter=linf_eps_iter)

        # self.attacker = GradientAttack(loss_fn=self.loss_fn, eps=l2_eps)
        # self.attacker = L2PGDAttack(
        #     loss_fn=self.loss_fn, num_iters=l2_niters, eps=l2_eps, eps_iter=l2_eps_iter)
        
    def forward(self, net, x, logit_y=None): 
        net.update_batch_stats(False)
        if logit_y is None:
            with torch.no_grad():
                logit_y = net(x)

        x_adv, _ = self.attacker.perturb(net, x, logit_y)
        net.update_batch_stats(True)
        
        logit_yadv = net(x_adv)
        loss = self.loss_fn(logit_yadv, logit_y.detach().clone())
        # loss = self.loss_fn(logit_y.detach().clone(), logit_yadv)
        return loss 
    
    
