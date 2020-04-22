
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from core.neural_net import ImgClassifierSmall, ImgClassifierLarge


class DeepModel(object):
    def __init__(self, args):
        self.device = args['device']
        self.num_epochs = args['num_epochs']
        self.vat_niters = args['vat_niters']
        self.vat_eps = args['vat_eps']
        self.vat_xi = args['vat_xi']
        
        # neuralnet and losses
        self.net = ImgClassifierSmall(args)
        self.cls_crit = nn.CrossEntropyLoss()

        self.net.to(self.device)
        self.cls_crit.to(self.device)
        
        # optimizer and lr scheduler
        self.optim = optim.Adam(self.net.parameters(), lr=args['lr'])

    def at_crit(self, x, y):

        x = x.detach().clone().requires_grad_(True)
        
        logit_y = self.net(x)
        ce = F.cross_entropy(logit_y, y)

        self.optim.zero_grad()
        ce.backward()

        noise = x.grad.detach()
        noise = normalize_vector(noise)

        r_adv = self.vat_eps*noise
        x_adv = x + r_adv
        logit_y = self.net(x_adv)
        ce = F.cross_entropy(logit_y, y)

        return ce
        
    def vat_crit(self, x, logit_y):
        # init noise
        noise = torch.empty(x.shape).normal_().to(self.device)
        noise = normalize_vector(noise)
        
        for i in range(self.vat_niters):
            noise = (self.vat_xi*noise).requires_grad_(True)
            x_noise = x + self.vat_xi*noise
            
            logit_yhat = self.net(x_noise)
            kl_div = compute_kl_div(logit_y.detach(), logit_yhat)
            
            self.optim.zero_grad()
            kl_div.backward()

            noise = noise.grad.detach()
            noise = normalize_vector(noise)

        r_adv = self.vat_eps*noise
        x_adv = x + r_adv
        logit_yhat = self.net(x_adv)
        kl_div = compute_kl_div(logit_y.detach(), logit_yhat)

        return kl_div
    
    def cem_crit(self, logit_y):
        py = F.softmax(logit_y, dim=1)
        return  -torch.mean(torch.sum(py*F.log_softmax(logit_y, dim=1), dim=1))

    def update(self, train_lbl, train_unlbl, epoch, logger):
        # switch net to train mode
        self.net.train()

        # make batch generators
        num_batches = min(len(train_lbl), len(train_unlbl))
        interval = int(num_batches/4) + 1
        train_lbl_iter = iter(train_lbl)
        train_unlbl_iter = iter(train_unlbl)

        # training
        train_loss = 0.
        start_time = time.time()
        for i in range(num_batches):

            # get train lbl and unlbl data
            lbl_x, lbl_y = train_lbl_iter.next()
            unlbl_x, _ = train_unlbl_iter.next()

            lbl_x = lbl_x.to(self.device)
            lbl_y = lbl_y.to(self.device)
            unlbl_x = unlbl_x.to(self.device)
            
            # feed data
            lbl_logit_y = self.net(lbl_x)
            unlbl_logit_y = self.net(unlbl_x)
            
            # compute losses
            lbl_loss_y = self.cls_crit(lbl_logit_y, lbl_y)
            
            lbl_loss_at = self.at_crit(lbl_x, lbl_y)
            lbl_loss_vat = self.vat_crit(lbl_x, lbl_logit_y)
            lbl_loss_cem = self.cem_crit(lbl_logit_y)
            
            unlbl_loss_vat = self.vat_crit(unlbl_x, unlbl_logit_y)
            unlbl_loss_cem = self.cem_crit(unlbl_logit_y)
            
            loss = lbl_loss_y + lbl_loss_at \
                + 0.5*(lbl_loss_vat + unlbl_loss_vat) \
                + 0.5*(lbl_loss_cem + unlbl_loss_cem)
            
            # backprop and update
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            # log info
            loss = loss.item()
            train_loss += loss

            i += 1            
            if i % interval == 0:
                logger.info('epoch %s  >> %3.2f (%2.3f sec) : loss = %2.5f'%(
                    epoch, (i+1)*100.0/num_batches, (time.time()-start_time), loss))
                
        logger.info('epoch %s  >> 100.00 (%2.3f sec) : train loss %2.5f'%(
            epoch, (time.time()-start_time), train_loss/num_batches))
        
        return train_loss/num_batches

    
    def evaluate(self, eval_lbl):
        # switch net to eval mode
        self.net.eval()
        
        # evaluating
        num_eval = len(eval_lbl.dataset)
        eval_y_corrects = 0.
        with torch.no_grad():
            for i, batch in enumerate(eval_lbl):
                eval_x, eval_y = batch
                eval_x = eval_x.to(self.device)
                eval_y = eval_y.to(self.device)

                eval_logit = self.net(eval_x)
                
                eval_pred = torch.argmax(eval_logit, dim=1)
                eval_y_corrects += torch.sum(eval_pred == eval_y).item()

        acc = eval_y_corrects/num_eval*100
        return {'acc': round(acc, 4), 'error': round(100-acc, 4)}


def normalize_vector(d):
    d_shape = d.shape
    d = d.view(d_shape[0], -1)
    # d /= (torch.max(torch.abs(d), dim=1, keepdim=True)[0] + 1e-12)
    d /= torch.sqrt(torch.sum(d**2, dim=1, keepdim=True) + 1e-6)
    d = d.view(d_shape)
    return d

def compute_kl_div(p_logit, q_logit):
    p_prob = F.softmax(p_logit, dim=1)
    diff = (F.log_softmax(p_logit, dim=1) - F.log_softmax(q_logit, dim=1))
    kl_div = torch.mean(torch.sum(p_prob * diff, dim=1))
    return kl_div
