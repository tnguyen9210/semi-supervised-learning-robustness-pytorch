
import time
import math 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.neural_net import LeNet5
from core.objectives import ATLoss
from core.transform import ImageTransform


class DeepModel(object):
    def __init__(self, args):
        self.device = args['device']
        self.num_iters_per_epoch = args['num_iters_per_epoch']
        self.it = 0
        
        # neuralnet and losses
        self.net = LeNet5(args)
        self.at_crit = ATLoss()

        self.net.to(self.device)
        
        # optimizer and lr scheduler
        self.optim = optim.SGD(
            self.net.parameters(), lr=args['lr'], momentum=args['momentum'],
            weight_decay=args['l2_params'])
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optim, args['scheduler_t0'], args['scheduler_tmult'])

    def update(self, train_lbl, epoch, logger):
        # switch net to train mode
        self.net.train()

        # make batch generators
        interval = self.num_iters_per_epoch // 4 + 1
        train_lbl_iter = iter(train_lbl)
        
        # training
        train_loss = 0.
        start_time = time.time()
        for i in range(self.num_iters_per_epoch):

            # get train lbl and unlbl data
            lbl_x, lbl_y = train_lbl_iter.next()

            lbl_x = lbl_x.to(self.device)
            lbl_y = lbl_y.to(self.device)
            
            # feed data
            # lbl_logit_y = self.net(lbl_x)

            # compute losses
            lbl_loss_y = self.at_crit(self.net, lbl_x, lbl_y)

            loss = lbl_loss_y
            
            # backprop and update
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            # log info
            loss = loss.item()
            train_loss += loss

            self.it += 1
            if (i + 1) % interval == 0:
                logger.info('epoch %s, it %s >> %3.2f (%2.3f sec) : loss = %2.5f'%(
                    epoch, (self.it), i*100.0/self.num_iters_per_epoch,
                    (time.time()-start_time), loss))

        train_loss = train_loss/self.num_iters_per_epoch
        
        lr = self.optim.param_groups[0]['lr']
        logger.info('epoch %s, it %s >> 100.00 (%2.3f sec) : lr %2.4f, train loss %2.5f'%(
            epoch, (self.it), (time.time()-start_time), lr, train_loss))
        
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

