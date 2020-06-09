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
    
    eval_lbl = ImageDataset(data_dir, domain, img_size, ds)

    eval_lbl = data.DataLoader(
        eval_lbl, batch_size=batch_size, shuffle=False, num_workers=16)
    
    return eval_lbl


def evaluate():
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

    # compute accuracy
    num_eval = len(eval_lbl.dataset)
    eval_corrects = 0.
    with torch.no_grad():
        for i, batch in enumerate(eval_lbl):
            # get batch data
            eval_x, eval_y = batch
            eval_x = eval_x.cuda()
            eval_y = eval_y.cuda()

            eval_logit = net(eval_x)
            eval_pred = torch.argmax(eval_logit, dim=1)

            eval_corrects += torch.sum(eval_pred == eval_y).item()

    acc = eval_corrects/num_eval*100
    print(f"{args['eval_set']}: error = {100-acc:2.4f}")
    
    
    
    
if __name__ == '__main__':
    evaluate()
