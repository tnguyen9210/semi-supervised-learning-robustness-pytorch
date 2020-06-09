
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import *


class VirtualGradientSignAttack(object):
    def __init__(self, loss_fn, eps, clip_min=0., clip_max=1.):

        self.loss_fn = loss_fn
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, net, x, logit_y=None):
        """
        """
        if logit_y is None:
            with torch.no_grad():
                logit_y = net(x)

                
        x_adv = x.detach().clone()
        logit_y = logit_y.detach().clone()

        # feed x_adv and compute grad
        x_adv.requires_grad = True
        logit_yadv = net(x_adv)
        loss = self.loss_fn(logit_yadv, logit_y)
        x_grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv + self.eps*x_grad.sign()
        x_adv = torch.clamp(x_adv, min=self.clip_min, max=self.clip_max)
        x_adv = x_adv.detach().clone()
        
        return x_adv, 0


class VirtualLinfPGDAttack(object):
    def __init__(
            self, loss_fn, num_iters, eps, eps_iter,
            rand_init=True, clip_min=0., clip_max=1.):

        self.loss_fn = loss_fn
        self.num_iters = num_iters
        self.eps = eps
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, net, x, logit_y=None):
        """
        """
        if logit_y is None:
            with torch.no_grad():
                logit_y = net(x)

                
        x_nat = x.detach().clone()
        logit_y = logit_y.detach().clone()
        
        # init perturb
        x_adv = x.detach().clone()
        if self.rand_init:
            delta = torch.zeros_like(x).uniform_(-1,1)
            delta = self.eps*delta
            x_adv = torch.clamp(x_adv + delta, min=self.clip_min, max=self.clip_max)

        # pgd iterations
        losses = []
        for it in range(self.num_iters):
            x_adv.requires_grad = True
            
            # feed x_adv and compute grad
            logit_yadv = net(x_adv)
            loss = self.loss_fn(logit_yadv, logit_y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            
            # compute delta
            x_adv = x_adv + self.eps_iter*grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_nat+self.eps), x_nat-self.eps)
            # x_adv = torch.clamp(x_adv, min=x_nat-self.eps, max=x_nat+self.eps)
            x_adv = torch.clamp(x_adv, min=self.clip_min, max=self.clip_max)
            x_adv = x_adv.detach().clone()

            losses.append(loss.item())
            
        return x_adv, losses

    
class VirtualGradientAttack(object):
    def __init__(self, loss_fn, eps, clip_min=0., clip_max=1.):

        self.loss_fn = loss_fn
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, net, x, logit_y=None):
        """
        """
        if logit_y is None:
            with torch.no_grad():
                logit_y = net(x)

                
        x_adv = x.detach().clone()
        logit_y = logit_y.detach().clone()

        # feed x_adv and compute grad
        x_adv.requires_grad = True
        logit_yadv = net(x_adv)
        loss = self.loss_fn(logit_yadv, logit_y)
        x_grad = torch.autograd.grad(loss, x_adv)[0]

        x_grad = normalize_by_pnorm(x_grad, 2)
        x_adv = x_adv + self.eps*x_grad
        x_adv = torch.clamp(x_adv, min=self.clip_min, max=self.clip_max)
        x_adv = x_adv.detach().clone()
        
        return x_adv, 0


class VirtualL2PGDAttack(object):
    def __init__(
            self, loss_fn, num_iters, eps, eps_iter,
            rand_init=True, clip_min=0., clip_max=1.):

        self.loss_fn = loss_fn
        self.num_iters = num_iters
        self.eps = eps
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, net, x, logit_y=None):
        """
        """
        if logit_y is None:
            with torch.no_grad():
                logit_y = net(x)
                
        x_nat = x.detach().clone()
        logit_y = logit_y.detach().clone()
        
        # init perturb
        x_adv = x.detach().clone()
        if self.rand_init:
            x_init = torch.zeros_like(x).uniform_(self.clip_min,self.clip_max)
            delta = x_init - x_adv
            delta = clamp_by_pnorm(delta, 2, self.eps)
            x_adv = torch.clamp(x_adv + delta, min=self.clip_min, max=self.clip_max)

        # pgd iterations
        losses = []
        for it in range(self.num_iters):
            x_adv.requires_grad = True
            
            # feed x_adv and compute grad
            logit_yadv = net(x_adv)
            loss = self.loss_fn(logit_yadv, logit_y)
            x_grad = torch.autograd.grad(loss, x_adv)[0]
            
            # compute delta
            x_grad = normalize_by_pnorm(x_grad, 2)
            x_adv = x_adv + self.eps_iter*x_grad
            x_adv = torch.max(torch.min(x_adv, x_nat+self.eps), x_nat-self.eps)
            # x_adv = torch.clamp(x_adv, min=x_nat-self.eps, max=x_nat+self.eps)
            x_adv = torch.clamp(x_adv, min=self.clip_min, max=self.clip_max)
            x_adv = x_adv.detach().clone()

            losses.append(loss.item())
            
        return x_adv, losses

