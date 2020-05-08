"""
Run evaluation with saved models
"""

import numpy as np

import torch 
import torch.utils.data as data

import torchvision.transforms as transforms

from core.deep_model import DeepModel

from dataset import ImageDataset
from config import parse_args
from utils import load_config


def load_data(data_dir, domain, ds, img_size, batch_size):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(), normalize])
    
    eval_lbl = ImageDataset(
        data_dir, domain, img_size, ds, transform=eval_transform)

    eval_lbl = data.DataLoader(
        eval_lbl, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return eval_lbl

def evaluate():
    args = parse_args()
    args = vars(args)
    # np.random.seed(args['seed'])
    # torch.manual_seed(args['seed'])
    # torch.cuda.manual_seed(args['seed'])
    
    model_dir = f"{args['model_dir']}/{args['model_id']}" 

    # load data
    eval_lbl = load_data(args['data_dir'], args['domain'], args['eval_set'],
                         args['img_size'], args['batch_size'])
    
    # load net
    model_args = load_config(model_dir)
    model = DeepModel(model_args)
    model.load_state(f"{model_dir}/{args['ckpt_name']}")
    # perfs = model.evaluate(eval_lbl)
    # print(f"{args['eval_set']}: error = {perfs['error']}")
    
    net = model.net
    net.eval()

    # compute accuracy
    num_eval = len(eval_lbl.dataset)
    eval_y_corrects = 0.
    with torch.no_grad():
        for i, batch in enumerate(eval_lbl):
            eval_x, eval_y = batch
            eval_x = eval_x.cuda()
            eval_y = eval_y.cuda()

            eval_logit = net(eval_x)
            eval_pred = torch.argmax(eval_logit, dim=1)

            eval_y_corrects += torch.sum(eval_pred == eval_y).item()

    acc = eval_y_corrects/num_eval*100
    print(f"{args['eval_set']}: error = {100-acc:2.4f}")
    
    
    
    
if __name__ == '__main__':
    evaluate()
