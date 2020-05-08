import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.sl_baseline.neural_net import ResNet


class DeepModel(object):
    def __init__(self, args):
        self.device = args['device']
        self.num_epochs = args['num_epochs']
        self.max_grad_norm = args['max_grad_norm']
        
        # neuralnet and losses
        self.net = ResNet(args)
        self.cls_crit = nn.CrossEntropyLoss()

        self.net.to(self.device)
        self.cls_crit.to(self.device)
        
        # optimizer and lr scheduler
        # self.optim = optim.Adam(self.net.parameters(), lr=args['lr'])
        self.optim = optim.SGD(
            self.net.parameters(), lr=args['lr'], momentum=args['momentum'],
            weight_decay=args['l2_params'])
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optim, args['scheduler_t0'], args['scheduler_tmult'])
        
    def update(self, train_lbl, epoch, logger):
        # switch net to train mode
        self.net.train()

        # make batch generators
        num_batches = len(train_lbl)
        interval = int(num_batches/4) + 1
        train_lbl_iter = iter(train_lbl)

        # training
        train_loss = 0.
        start_time = time.time()
        for i in range(num_batches):

            # get train lbl data
            train_lbl_x, train_lbl_y = train_lbl_iter.next()

            train_lbl_x = train_lbl_x.to(self.device)
            train_lbl_y = train_lbl_y.to(self.device)
            
            # feed data
            train_lbl_logit_y = self.net(train_lbl_x)
            
            # compute los
            loss = self.cls_crit(train_lbl_logit_y, train_lbl_y)
            
            # backprop and update
            self.optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optim.step()
            
            # log info
            loss = loss.item()
            train_loss += loss

            i += 1            
            if i % interval == 0:
                logger.info('epoch %s  >> %3.2f (%2.3f sec) : loss = %2.5f'%(
                    epoch, (i+1)*100.0/num_batches, (time.time()-start_time), loss))

        lr = self.optim.param_groups[0]['lr']
        logger.info('epoch %s  >> 100.00 (%2.3f sec) : lr %2.4f, train loss %2.5f'%(
            epoch, (time.time()-start_time), lr, train_loss/num_batches))

        self.lr_scheduler.step()
        
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
