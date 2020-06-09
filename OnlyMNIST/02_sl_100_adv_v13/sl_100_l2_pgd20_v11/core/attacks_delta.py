
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import *


class GradientSignAttack(object):
    def __init__(self, loss_fn, eps, clip_min=0., clip_max=1.):

        self.loss_fn = loss_fn
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, net, x, y=None):
        """
        """
        if y is None:
            with torch.no_grad():
                logit = net(x)
            y = torch.argmax(logit, dim=1)
            
        x_adv = x.detach().clone()
        y = y.detach().clone()

        # feed x_adv and compute grad
        x_adv.requires_grad = True
        logit_yadv = net(x_adv)
        loss = self.loss_fn(logit_yadv, y)
        x_grad = torch.autograd.grad(loss, x_adv)[0]
        
        x_adv = x_adv + self.eps*x_grad.sign()
        x_adv = torch.clamp(x_adv, min=self.clip_min, max=self.clip_max)
        x_adv = x_adv.detach().clone()
        
        return x_adv, 0

    
class LinfPGDAttack(object):
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

    def perturb(self, net, x, y=None):
        """
        """
        if y is None:
            with torch.no_grad():
                logit = net(x)
            y = torch.argmax(logit, dim=1)
            
        x_nat = x.detach().clone()
        y = y.detach().clone()
        
        # init perturb
        if self.rand_init:
            delta = torch.zeros_like(x).uniform_(-1,1)
            delta = self.eps*delta
            x_adv = torch.clamp(x_nat + delta, min=self.clip_min, max=self.clip_max)
            delta = (x_adv - x_nat).detach().clone()

        # pgd iterations
        losses = []
        for it in range(self.num_iters):
            delta.requires_grad = True
            
            # feed x_adv and compute grad
            x_adv = x_nat + delta
            logit_yadv = net(x_adv)
            loss = self.loss_fn(logit_yadv, y)
            grad = torch.autograd.grad(loss, delta)[0]
            
            # compute delta
            delta = delta + self.eps_iter*grad.sign()
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x_nat+delta, min=self.clip_min, max=self.clip_max)
            delta = (x_adv-x_nat).detach().clone()
            
            losses.append(round(loss.item(), 4))

        x_adv = x_nat + delta
        
        return x_adv, losses

    
class GradientAttack(object):
    def __init__(self, loss_fn, eps, clip_min=0., clip_max=1.):

        self.loss_fn = loss_fn
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, net, x, y=None):
        """
        """
        if y is None:
            with torch.no_grad():
                logit = net(x)
            y = torch.argmax(logit, dim=1)

        x_adv = x.detach().clone()
        y = y.detach().clone()

        # feed x_adv and compute grad
        x_adv.requires_grad = True
        logit_yadv = net(x_adv)
        loss = self.loss_fn(logit_yadv, y)
        x_grad = torch.autograd.grad(loss, x_adv)[0]

        x_grad = normalize_by_pnorm(x_grad, 2)
        x_adv = x_adv + self.eps*x_grad
        x_adv = torch.clamp(x_adv, min=self.clip_min, max=self.clip_max)
        x_adv = x_adv.detach().clone()
        
        return x_adv, 0


class L2PGDAttack(object):
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

    def perturb(self, net, x, y=None):
        """
        """
        if y is None:
            with torch.no_grad():
                logit = net(x)
            y = torch.argmax(logit, dim=1)
            
        x_nat = x.detach().clone()
        y = y.detach().clone()
        
        # init perturb
        if self.rand_init:
            x_adv = torch.zeros_like(x).uniform_(self.clip_min,self.clip_max)
            delta = x_adv - x_nat
            delta = clamp_by_pnorm(delta, 2, self.eps)
            x_adv = torch.clamp(x_nat + delta, min=self.clip_min, max=self.clip_max)
            delta = (x_adv - x_nat).detach().clone()

        # pgd iterations
        losses = []
        for it in range(self.num_iters):
            delta.requires_grad = True
            
            # feed x_adv and compute grad
            x_adv = x_nat + delta
            logit_yadv = net(x_adv)
            loss = self.loss_fn(logit_yadv, y)
            grad = torch.autograd.grad(loss, delta)[0]
            
            # compute delta
            grad = normalize_by_pnorm(grad, 2)
            delta = delta + self.eps_iter*grad
            delta = clamp_by_pnorm(delta, 2, self.eps)
            x_adv = torch.clamp(x_nat+delta, min=self.clip_min, max=self.clip_max)
            delta = (x_adv-x_nat).detach().clone()
            
            losses.append(round(loss.item(), 4))

        x_adv = x_nat + delta
        
        return x_adv, losses
