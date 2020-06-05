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


def load_data(data_dir, domain, ds, img_size, batch_size):

    normalize = transforms.Normalize(
            mean=[0.4914, 0.4821, 0.4463], std=[0.2467, 0.2431, 0.2611])
    
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    eval_lbl = ImageDataset(data_dir, domain, img_size, ds, transform=test_transform)

    eval_lbl = data.DataLoader(
        eval_lbl, batch_size=batch_size, shuffle=False, num_workers=16)
    
    return eval_lbl


def test_attacks():
    args = parse_args()
    args = vars(args)
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])

    # dir
    model_dir = f"{args['model_dir']}/{args['model_id']}" 

    # load data
    eval_lbl = load_data(args['data_dir'], args['domain'], args['eval_set'],
                         args['img_size'], 200)
    
    # load net
    model_args = load_config(model_dir)
    model = DeepModel(model_args)
    model.load_state(f"{model_dir}/{args['ckpt_name']}")

    net = model.net
    net.eval()

    # load attacker
    linf_eps = 0.3
    linf_eps_iter = 0.01
    linf_niters = 40
    l2_eps = 3.
    l2_eps_iter = 0.5
    l2_niters = 40
    
    # generic
    attackers = {
        'linf_fgsm': attacks.GradientSignAttack(net, loss_fn=F.cross_entropy, eps=linf_eps),
        'linf_pgd40': attacks.LinfPGDAttack(
            net, loss_fn=F.cross_entropy, eps=linf_eps, nb_iter=linf_niters, eps_iter=linf_eps_iter),
        
        'l2_fgm': attacks.GradientAttack(net, loss_fn=F.cross_entropy, eps=l2_eps),
        'l2_pgd40': attacks.L2PGDAttack(
            net, loss_fn=F.cross_entropy, eps=l2_eps, nb_iter=l2_niters, eps_iter=l2_eps_iter),        
    }

    # compute orig and adv accuracy
    for name, attacker in attackers.items():
        num_eval = len(eval_lbl.dataset)
        all_orig_x = []
        all_orig_y = []
        all_adv_x = []
        for i, batch in enumerate(eval_lbl):
            # get batch data 
            orig_x, orig_y = batch
            orig_x = orig_x.cuda()
            orig_y = orig_y.cuda()

            # generate adv samples
            adv_x = attacker.perturb(orig_x, orig_y)

            all_orig_x.append(orig_x.cpu())
            all_orig_y.append(orig_y.cpu())
            all_adv_x.append(adv_x.detach().cpu())
            
        all_orig_x = torch.cat(all_orig_x, axis=0)
        all_orig_y = torch.cat(all_orig_y, axis=0)
        all_adv_x = torch.cat(all_adv_x, axis=0)
        print(all_orig_x.shape)
        print(all_adv_x.shape)
        torch.save((all_orig_x, all_orig_y, all_adv_x), f"./data_adv/{name}.pt")
    
if __name__ == '__main__':
    test_attacks()
