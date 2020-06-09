
"""
Test adversarial attacks with saved models
"""

import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn.functional as F
import torch.utils.data as data

from torchvision import transforms, utils
from advertorch.attacks import GradientSignAttack, CarliniWagnerL2Attack, PGDAttack

from core.deep_model import DeepModel

from dataset import ImageDataset
from config import parse_args
from utils import load_config


def load_data(data_dir, domain, ds, img_size, batch_size):

    normalize = transforms.Normalize(
      mean=[0.4914, 0.4821, 0.4463], std=[0.2467, 0.2431, 0.2611])

    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        #  normalize
    ])
    
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
                         args['img_size'], 8)
    
    # load net
    model_args = load_config(model_dir)
    model = DeepModel(model_args)
    model.load_state(f"{model_dir}/{args['ckpt_name']}")
    
    net = model.net
    net.eval()

    # load attacker
    # attacker = GradientSignAttack(net, loss_fn=F.cross_entropy, eps=8.0/255)
    attacker = PGDAttack(
        net, loss_fn=F.cross_entropy, eps=8.0/255., nb_iter=7, eps_iter=2.0/255.)

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

    # generate adv samples
    adv_x = attacker.perturb(orig_x, orig_y)

    # compute adv accuracy
    adv_logit = net(adv_x)
    adv_pred = torch.argmax(adv_logit, dim=1)
        
    # show examples with 
    # show_sample_images(orig_x.cpu(), orig_y.cpu(), orig_pred, title='orig')
    show_sample_images(orig_x.cpu(), adv_x.detach().cpu(), orig_y.cpu(), orig_pred, adv_pred)
    # dump_sample_images(orig_x.cpu(), adv_x.detach().cpu(), orig_y.cpu(), orig_pred, adv_pred)

def dump_sample_images(orig_images, adv_images, orig_labels, orig_preds, adv_preds):
    from PIL import Image
    print(f"--> orig labels: {orig_labels}")
    print(f"--> orig preds: {orig_preds}")
    print(f"--> adv preds: {adv_preds}")

    idx = 0
    for oi, ai, ol, op, ap in zip(orig_images, adv_images, orig_labels, orig_preds, adv_preds):
      oi = oi.numpy().transpose((1, 2, 0))
      oi = np.clip(oi, 0, 1) * 255
      oi = oi.astype('uint8')
      ai = ai.numpy().transpose((1, 2, 0))
      ai = np.clip(ai, 0, 1) * 255
      ai = ai.astype('uint8')

      oi = Image.fromarray(oi)
      ai = Image.fromarray(ai)

      oi.save("oi_{}.jpeg".format(idx))
      ai.save("ai_{}.jpeg".format(idx))
      idx += 1
    
def show_sample_images(orig_images, adv_images, orig_labels, orig_preds, adv_preds):
    print(f"--> orig labels: {orig_labels}")
    print(f"--> orig preds: {orig_preds}")
    print(f"--> adv preds: {adv_preds}")

    fig, axes = plt.subplots(2, 1)
    axes = axes.flatten()
        
    orig_images = utils.make_grid(orig_images, padding=10)
    orig_images = orig_images.numpy().transpose((1, 2, 0))
    orig_images = np.clip(orig_images, 0, 1)
    axes[0].imshow(orig_images)

    adv_images = utils.make_grid(adv_images, padding=10)
    adv_images = adv_images.numpy().transpose((1, 2, 0))
    adv_images = np.clip(adv_images, 0, 1)
    axes[1].imshow(adv_images)
    
    plt.show()
    plt.savefig('temp.png')
 
if __name__ == '__main__':
    test_attacks()
