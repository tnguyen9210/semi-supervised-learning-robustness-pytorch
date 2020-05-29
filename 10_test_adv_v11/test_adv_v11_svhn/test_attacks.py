"""
Test adversarial attacks with saved models
"""

import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data

import torchvision.transforms as transforms
from advertorch import attacks 
#import GradientSignAttack, CarliniWagnerL2Attack, PGDAttack

from core.deep_model import DeepModel

from dataset import ImageDataset
from config import parse_args
from utils import load_config


def load_data(data_dir, domain, ds, img_size, batch_size):
    
    eval_lbl = ImageDataset(data_dir, domain, img_size, ds)

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
                         args['img_size'], args['batch_size'])

    
    # load net
    model_args = load_config(model_dir)
    model = DeepModel(model_args)
    model.load_state(f"{model_dir}/{args['ckpt_name']}")
    
    net = model.net
    net.eval()

    # compute orig and adv accuracy
    num_eval = len(eval_lbl.dataset)
    orig_corrects = 0.
    adv_corrects = 0.
    for i, batch in enumerate(eval_lbl):
        # get batch data 
        orig_x, orig_y = batch
        orig_x = orig_x.cuda()
        orig_y = orig_y.cuda()
        
        # compute orig accuracy
        orig_logit = net(orig_x)
        orig_pred = torch.argmax(orig_logit, dim=1)
        orig_corrects += torch.sum(orig_pred == orig_y).item()

    orig_acc = orig_corrects/num_eval*100
    adv_acc = adv_corrects/num_eval*100
    print(f"{args['eval_set']}: orig acc = {orig_acc:2.4f}")

    # common config for attacks
    common_attack_params = {
        "loss_fn": F.cross_entropy,
        "eps": 0.3,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "targeted": False,
    }

    num_classes = 10

    # load attacker
    attackers = {
        'GradientSignAttack': attacks.GradientSignAttack(
            net, **common_attack_params),
        #'PGD': attacks.PGDAttack(
        #    net, **common_attack_params, nb_iter=40, eps_iter=0.01),
        'LinfPGDAttack': attacks.LinfPGDAttack(
            net, **common_attack_params, nb_iter=40, eps_iter=0.01),
        'L2PGDAttack': attacks.L2PGDAttack(
            net, **common_attack_params, nb_iter=40, eps_iter=0.01),
        #'CarlinWagnerL2Attack': attacks.CarliniWagnerL2Attack(
        #    net, num_classes, loss_fn=F.cross_entropy), # Use default values. Not familiar with implementation
        }

    for name, attacker in attackers.items():
      print("Running {} attack...".format(name))

      # compute orig and adv accuracy
      num_eval = len(eval_lbl.dataset)
      adv_corrects = 0.
      for i, batch in enumerate(eval_lbl):
          # get batch data 
          orig_x, orig_y = batch
          orig_x = orig_x.cuda()
          orig_y = orig_y.cuda()
          
          # generate adv samples
          adv_x = attacker.perturb(orig_x, orig_y)

          # compute adv accuracy
          adv_logit = net(adv_x)
          adv_pred = torch.argmax(adv_logit, dim=1)
          adv_corrects += torch.sum(adv_pred == orig_y).item()
          
      adv_acc = adv_corrects/num_eval*100
      print(f"{name} attack adv acc = {adv_acc:2.4f}")
    
if __name__ == '__main__':
    test_attacks()
