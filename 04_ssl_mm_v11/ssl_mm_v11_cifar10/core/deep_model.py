
import time
import math 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.neural_net import ResNet
from core.objectives import MixMatch
from core.transform import ImageTransform


class DeepModel(object):
    def __init__(self, args):
        self.device = args['device']
        self.num_iters_per_epoch = args['num_iters_per_epoch']
        self.mm_num_augments = args['mm_num_augments']
        self.mm_temperature = args['mm_temperature']
        self.mm_alpha = args['mm_alpha']
        self.consis_coef = args['mm_consis_coef']
        self.consis_warmup = args['consis_warmup']
        self.it = 0
        
        # neuralnet and losses
        self.net = ResNet(args)
        self.transform = ImageTransform(random_horizontal_flip=True, random_crop=True)
        self.mix_match = MixMatch(
            self.mm_num_augments, self.mm_temperature, self.mm_alpha)

        self.net.to(self.device)
        
        # optimizer and lr scheduler
        # self.optim = optim.Adam(self.net.parameters(), lr=args['lr'])
        self.optim = optim.SGD(
            self.net.parameters(), lr=args['lr'], momentum=args['momentum'],
            weight_decay=args['l2_params'])
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optim, args['scheduler_t0'], args['scheduler_tmult'])

    def update(self, train_lbl, train_unlbl, epoch, logger):
        # switch net to train mode
        self.net.train()

        # make batch generators
        interval = self.num_iters_per_epoch // 4 + 1
        train_lbl_iter = iter(train_lbl)
        train_unlbl_iter = iter(train_unlbl)
        
        # training
        train_loss = 0.
        start_time = time.time()
        for i in range(self.num_iters_per_epoch):
            self.it += 1

            # get train lbl and unlbl data
            lbl_x, lbl_y = train_lbl_iter.next()
            unlbl_x, _ = train_unlbl_iter.next()

            # transform lbl_y into one-hot (prob)
            lbl_y = torch.zeros(lbl_x.shape[0], 10).scatter_(1, lbl_y.view(-1,1), 1)

            lbl_x = lbl_x.to(self.device)
            lbl_y = lbl_y.to(self.device)
            unlbl_x = unlbl_x.to(self.device)
            
            # mix-match lbl and unlbl batches
            unlbl_logit, unlbl_yhat, lbl_logit, lbl_yhat = self.mix_match(
                unlbl_x, lbl_x, lbl_y, self.net, self.transform)

            # compute loss
            lbl_loss = -torch.mean(torch.sum(
                F.log_softmax(lbl_logit, dim=1)*lbl_yhat, dim=1))
            
            unlbl_prob = torch.softmax(unlbl_logit, dim=1)
            unlbl_loss = torch.mean((unlbl_prob - unlbl_yhat)**2)
            
            # ramp up exp(-5(1 - t)^2)
            coef = self.consis_coef \
                * math.exp(-5*(1 - min(self.it/self.consis_warmup, 1))**2)
            # coef = self.consis_coef
            loss = lbl_loss + coef*unlbl_loss
            
            # backprop and update
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            # log info
            loss = loss.item()
            train_loss += loss

            i += 1            
            if i % interval == 0:
                logger.info('epoch %s, it %s >> %3.2f (%2.3f sec) : loss = %2.5f'%(
                    epoch, (self.it), i*100.0/self.num_iters_per_epoch,
                    (time.time()-start_time), loss))

        train_loss = train_loss/self.num_iters_per_epoch
        
        lr = self.optim.param_groups[0]['lr']
        logger.info('epoch %s  >> 100.00 (%2.3f sec) : lr %2.4f, train loss %2.5f'%(
            epoch, (time.time()-start_time), lr, train_loss))
        
        self.lr_scheduler.step()
        
        return train_loss

    
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

    def load_state(self, filename):
        state_dict = torch.load(filename, map_location='cuda:0')
        self.net.load_state_dict(state_dict)
        
    def save_state(self, save_dir, epoch=None):
        if epoch:
            save_file = f"{save_dir}/epoch_{epoch}.ckpt"
        else:
            save_file = f"{save_dir}/best_model.ckpt"
        # prevent disruption during saving
        try:
            torch.save(self.net.state_dict(), save_file)
            print("model saved to {}".format(save_file))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

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
