"""
Usage: python generate.py
Description: Generates adversarial attacks based on advertorch
"""

from time import time
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from advertorch import attacks
from utils import baseline_data_loader, load_sl_baseline_model


"""
Loads (pretrained) pytorch model
"""
def load_model(path_to_ckpt=None):
  # Resnet for now
  model = models.resnet18(pretrained=True)
  return model


"""
Dumps perturbed inputs to pkl files  
"""
def dump_to_pkl(model_name, data, labels, type_of_attack=None, idx_offset = 0):

  # Keeping track of the index
  idx = idx_offset

  assert len(data) == len(labels), "data and labels should be same length: data:{}, labels:{}".format(data.shape[0], labels.shape[0])
    
  for img, label in zip(data, labels):
    file_path = os.path.join('x_adv', model_name, type_of_attack, str(label))
    if not os.path.exists(file_path):
      os.makedirs(file_path)

    # Generate pkl file in path <type_of_attack>/<label>/***.pkl
    file_name = "{}.pkl".format(os.path.join(file_path, str(idx)))
    img = np.array(img)
    img = np.moveaxis(img, 0, -1)
    with open(file_name, "wb") as f:
      pickle.dump(img, f)
    idx += 1

"""
Generates perturbed adversarial inputs
"""
def generate_gradient_sign_attack(model, data_loader, eps=0.01):
  tick = time()

  model = model.to(device)
  model.eval()

  print("Generating Inputs for Gradient Sign Attack...")

  adversary = attacks.GradientSignAttack(
      model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, 
      clip_min=0.0, clip_max=1.0, targeted=False
      )

  cnt = 0
  for data, labels in data_loader:
    data = data.to(device)
    labels = labels.to(device)

    adv_untargeted = adversary.perturb(data, labels)
    print("Saving x_adv for {} ~ {}".format(cnt, cnt + len(data)))
    dump_to_pkl(
        os.path.join(model.name, model.dataset), 
        adv_untargeted.tolist(), labels.tolist(), 
        type_of_attack='GradientSignAttack', idx_offset=cnt)
    cnt += len(data)

  time_elapsed = time() - tick
  print('GradientSignAttack generation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def generate_pgd_attack(model, data_loader, eps=0.01):
  tick = time()

  model = model.to(device)
  model.eval()

  print("Generating Inputs for PGD(Madry) Attack...")

  adversary = attacks.PGDAttack(
      model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, 
      nb_iter=40, eps_iter=0.01, clip_min=0.0, clip_max=1.0, targeted=False
      )

  cnt = 0
  for data, labels in data_loader:
    data = data.to(device)
    labels = labels.to(device)

    adv_untargeted = adversary.perturb(data, labels)
    print("Saving x_adv for {} ~ {}".format(cnt, cnt + len(data)))
    dump_to_pkl(
        os.path.join(model.name, model.dataset), 
        adv_untargeted.tolist(), labels.tolist(), 
        type_of_attack='PGDAttack', idx_offset=cnt)
    cnt += len(data)

  time_elapsed = time() - tick
  print('PGDAttack generation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def generate_linf_pgd_attack(model, data_loader, eps=0.01):
  tick = time()

  model = model.to(device)
  model.eval()

  print("Generating Inputs for L_inf PGD Attack...")

  adversary = attacks.LinfPGDAttack(
      model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, 
      nb_iter=40, eps_iter=0.01, clip_min=0.0, clip_max=1.0, targeted=False
      )

  cnt = 0
  for data, labels in data_loader:
    data = data.to(device)
    labels = labels.to(device)

    adv_untargeted = adversary.perturb(data, labels)
    print("Saving x_adv for {} ~ {}".format(cnt, cnt + len(data)))
    dump_to_pkl(
        os.path.join(model.name, model.dataset), 
        adv_untargeted.tolist(), labels.tolist(), 
        type_of_attack='LinfPGDAttack', idx_offset=cnt)
    cnt += len(data)

  time_elapsed = time() - tick
  print('LinfPGDAttack generation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def generate_l2_pgd_attack(model, data_loader, eps=0.01):
  tick = time()

  model = model.to(device)
  model.eval()

  print("Generating Inputs for L2 PGD Attack...")

  adversary = attacks.L2PGDAttack(
      model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, 
      nb_iter=40, eps_iter=0.01, clip_min=0.0, clip_max=1.0, targeted=False
      )

  cnt = 0
  for data, labels in data_loader:
    data = data.to(device)
    labels = labels.to(device)

    adv_untargeted = adversary.perturb(data, labels)
    print("Saving x_adv for {} ~ {}".format(cnt, cnt + len(data)))
    dump_to_pkl(
        os.path.join(model.name, model.dataset), 
        adv_untargeted.tolist(), labels.tolist(), 
        type_of_attack='L2PGDAttack', idx_offset=cnt)
    cnt += len(data)

  time_elapsed = time() - tick
  print('L2PGDAttack generation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def generate_l1_pgd_attack(model, data_loader, eps=0.01):
  tick = time()

  model = model.to(device)
  model.eval()

  print("Generating Inputs for L1 PGD Attack...")

  adversary = attacks.L1PGDAttack(
      model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, 
      nb_iter=40, eps_iter=0.01, clip_min=0.0, clip_max=1.0, targeted=False
      )

  cnt = 0
  for data, labels in data_loader:
    data = data.to(device)
    labels = labels.to(device)

    adv_untargeted = adversary.perturb(data, labels)
    print("Saving x_adv for {} ~ {}".format(cnt, cnt + len(data)))
    dump_to_pkl(
        os.path.join(model.name, model.dataset), 
        adv_untargeted.tolist(), labels.tolist(), 
        type_of_attack='L1PGDAttack', idx_offset=cnt)
    cnt += len(data)

  time_elapsed = time() - tick
  print('L1PGDAttack generation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":
  print("==============================")
  print("Need to (hard-code) specify the following:")
  print("==============================")
  print("Path to model, default model is Resnet-18")
  print("Types of adversarial attacks, default attack is Gradient Sign Attack")
  print("Type of dataset, default is CIFAR-10")
  print("Path to write adversarial examples, default is './x_adv/<model_name>/<dataset>/<type_of_attack>/<image_id>.pkl'")

  # Parse arguments
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default="cifar10", help="cifar10 | svhn")
  parser.add_argument('--sl_or_ssl', type=str, default="sl", help="sl | ssl")
  parser.add_argument('--base_or_vat', type=str, default="base", help="base | vat")
  args =  parser.parse_args()

  args = vars(args)
  dataset = args['dataset']
  sl_or_ssl = args['sl_or_ssl']
  base_or_vat = args['base_or_vat']

  model_name = "{}_{}_{}".format(sl_or_ssl, base_or_vat, dataset)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("Generating using: {}".format(device))

  if dataset == "cifar10":
    # Load test data
    data_loader = baseline_data_loader("CIFAR10")
  elif dataset == "svhn":
    data_loader = baseline_data_loader("SVHN")

  # Load model
  model = load_sl_baseline_model('pretrained/{}_{}_v11_{}'.format(sl_or_ssl, base_or_vat, dataset), with_weights=True)
  model = model.net
  model.name = model_name
  model.dataset = dataset

  # Generate pertubed data
  generate_gradient_sign_attack(model, data_loader)
  generate_pgd_attack(model, data_loader)
  generate_linf_pgd_attack(model, data_loader)
  generate_l2_pgd_attack(model, data_loader)
  # Implementation has a minor bug, soon will be fixed in advertorch repo
  # generate_l1_pgd_attack(model, data_loader)
