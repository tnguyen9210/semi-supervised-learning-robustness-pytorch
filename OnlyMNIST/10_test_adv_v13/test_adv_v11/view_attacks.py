"""
Test adversarial attacks with saved models
"""
import matplotlib.pyplot as plt 
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data

from torchvision import transforms, utils, datasets

import advertorch.attacks as attacks
from core.attacks import *
from core.attacks_virtual import *

from core.deep_model import DeepModel

from dataset import ImageDataset
from config import parse_args
from utils import load_config


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
    eval_lbl = load_data(
        args['data_dir'], args['domain'], args['eval_set'], args['img_size'], 8)
    
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
    l2_eps = 4.0
    l2_eps_iter = 0.25
    l2_niters = 40
    
    # attacker = attacks.GradientSignAttack(net, loss_fn=F.cross_entropy, eps=linf_eps)
    attacker = attacks.LinfPGDAttack(
            net, loss_fn=F.cross_entropy, eps=linf_eps, nb_iter=linf_niters, eps_iter=linf_eps_iter)

    # attacker = attacks.GradientAttack(net, loss_fn=F.cross_entropy, eps=l2_eps)
    # attacker = attacks.L2PGDAttack(
    #         net, loss_fn=F.cross_entropy, eps=l2_eps, nb_iter=l2_niters, eps_iter=l2_eps_iter)

    # attacker = VirtualGradientSignAttack(loss_fn=kl_div, eps=linf_eps)
    # attacker = VirtualLinfPGDAttack(
    #     loss_fn=kl_div, num_iters=7, eps=linf_eps, eps_iter=linf_eps_iter)
        
    
    # compute orig and adv accuracy
    num_eval = len(eval_lbl.dataset)
    
    # get batch data
    eval_lbl_iter = iter(eval_lbl)
    orig_x, orig_y = eval_lbl_iter.next()
    orig_x = orig_x.cuda()
    orig_y = orig_y.cuda()
    
    # compute orig accuracy
    orig_logit = net(orig_x)
    orig_pred = torch.argmax(orig_logit, dim=1)
    orig_corrects = torch.sum(orig_pred == orig_y).item()

    # generate adv samples
    adv_x = attacker.perturb(orig_x, orig_y)
    # adv_x, losses = attacker.perturb(net, orig_x)

    # compute adv accuracy
    adv_logit = net(adv_x)
    adv_pred = torch.argmax(adv_logit, dim=1)
    adv_corrects = torch.sum(adv_pred == orig_y).item()
    
    print(f"{args['eval_set']}: "
          f"orig corrects = {orig_corrects:2d}, adv acc = {adv_corrects:2d}")

    # show examples with 
    show_sample_images(orig_x.cpu(), adv_x.detach().cpu(), orig_y.cpu(), orig_pred, adv_pred)

def show_sample_images(orig_images, adv_images, orig_labels, orig_preds, adv_preds):
    print(f"--> orig labels: {orig_labels}")
    print(f"--> orig preds: {orig_preds}")
    print(f"--> adv preds: {adv_preds}")

    fig, axes = plt.subplots(2, 1)
    axes = axes.flatten()
        
    orig_images = utils.make_grid(orig_images, padding=2, normalize=True)
    orig_images = orig_images.numpy().transpose((1, 2, 0))
    orig_images = np.clip(orig_images, 0, 1)
    axes[0].imshow(orig_images)

    adv_images = utils.make_grid(adv_images, padding=2, normalize=True)
    adv_images = adv_images.numpy().transpose((1, 2, 0))
    adv_images = np.clip(adv_images, 0, 1)
    axes[1].imshow(adv_images)
    
    plt.show()

if __name__ == '__main__':
    test_attacks()
