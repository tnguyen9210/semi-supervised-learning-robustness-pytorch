
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data

import torchvision.transforms as transforms

from core.deep_model import DeepModel

from dataset import ImageDataset
from config import parse_args
from utils import load_config


def load_data(data_dir, domain, ds, img_size, batch_size):
    normalize = transforms.Normalize(
            mean=[0.4377, 0.4438, 0.4728], std=[0.1975, 0.2004, 0.1964])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(), normalize])
    
    eval_lbl = ImageDataset(
        data_dir, domain, img_size, ds, transform=eval_transform)

    eval_lbl = data.DataLoader(
        eval_lbl, batch_size=1, shuffle=False)
    
    return eval_lbl

def test_attacks():
    args = parse_args()
    args = vars(args)
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    
    # params
    adv_eps = 0.1

    # load data
    eval_lbl = load_data(args['data_dir'], args['domain'], args['eval_set'],
                         args['img_size'], batch_size=1)
    
    # load net
    model_dir = f"{args['model_dir']}/{args['model_id']}"
    model_ckpt = f"{args['model_dir']}/{args['model_id']}/{args['ckpt_name']}"
    model_args = load_config(model_dir)
    model = DeepModel(model_args)
    model.load_state(model_ckpt)
    net = model.net
    net.eval()

    # test attacks
    num_eval = len(eval_lbl.dataset)
    adv_corrects = 0.
    for i, (orig_img, label) in enumerate(eval_lbl):
        # if i > 1000:
        #     break
        
        # get orig img and label
        label = label.cuda()
        orig_img = orig_img.cuda()
        orig_img.requires_grad_(True)
        
        # feed img and compute pred
        orig_logit = net(orig_img)
        orig_pred = torch.argmax(orig_logit, dim=1)

        # if orig prediction is wrong, do not bother attacking
        if orig_pred != label:
            continue

        net.zero_grad()
        
        # compute grad
        loss = F.cross_entropy(orig_logit, label)
        loss.backward()
        grad = orig_img.grad.detach()

        # compute FGSM attack
        fgsm_grad = grad.sign()
        adv_img = orig_img + adv_eps*fgsm_grad
        adv_img = torch.clamp(adv_img, 0, 1)  # clipping to maintain [0, 1] range
        
        # test on adv image
        adv_logit = net(adv_img)
        adv_pred = torch.argmax(adv_logit, dim=1)
        if adv_pred == label:
            adv_corrects += 1

    acc = adv_corrects/num_eval*100
    print(f"{args['eval_set']}: error = {100-acc:2.4f}")
    
    
if __name__ == '__main__':
    test_attacks()
