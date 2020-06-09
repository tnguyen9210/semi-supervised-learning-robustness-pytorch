"""
Test adversarial attacks with saved models
"""
import matplotlib.pyplot as plt 
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data

import torchvision.transforms as transforms

import advertorch.attacks as attacks
# from core.attacks import *
from core.attacks_delta import *
from core.attacks_virtual import *

from core.deep_model import DeepModel

from dataset import ImageDataset
from config import parse_args
from utils import load_config, print_config


class ImageDatasetAdv(data.Dataset):
    def __init__(self, attacker):
        self.orig_x, self.orig_y, self.adv_x = \
            torch.load(f"./data_adv/{attacker}.pt")
        
        self.num_data = self.orig_x.shape[0]
        
    def __getitem__(self, idx):
        return self.orig_x[idx], self.orig_y[idx], self.adv_x[idx]
    
    def __len__(self):
        return self.num_data



def test_attacks():
    args = parse_args()
    args = vars(args)
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])

    # dir
    model_dir = f"{args['model_dir']}/{args['model_id']}" 

    # load net
    model_args = load_config(model_dir)
    model = DeepModel(model_args)
    model.load_state(f"{model_dir}/{args['ckpt_name']}")

    net = model.net
    net.eval()

    # generic
    attackers = ['linf_fgsm', 'linf_pgd40', 'l2_fgm', 'l2_pgd40']
    
    # compute orig and adv accuracy
    for attacker in attackers:
        eval_lbl = data.DataLoader(
            ImageDatasetAdv(attacker), batch_size=200, shuffle=False, num_workers=16)
        num_eval = len(eval_lbl.dataset)
        orig_corrects = 0.
        adv_corrects = 0.
        for i, batch in enumerate(eval_lbl):
            # get batch data 
            orig_x, orig_y, adv_x = batch
            
            orig_x = orig_x.cuda()
            orig_y = orig_y.cuda()
            adv_x = adv_x.cuda()

            # compute orig accuracy
            orig_logit = net(orig_x)
            orig_pred = torch.argmax(orig_logit, dim=1)
            orig_corrects += torch.sum(orig_pred == orig_y).item()

            # compute adv accuracy
            adv_logit = net(adv_x)
            adv_pred = torch.argmax(adv_logit, dim=1)
            adv_corrects += torch.sum(adv_pred == orig_y).item()
        
        orig_acc = orig_corrects/num_eval*100
        adv_acc = adv_corrects/num_eval*100
        print(f"{args['eval_set']}: orig acc = {orig_acc:2.4f}, adv acc = {adv_acc:2.4f}")

    
if __name__ == '__main__':
    test_attacks()
