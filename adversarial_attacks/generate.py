"""
Usage: python generate.py
Description: Generates adversarial attacks based on advertorch
"""

from time import time
import os
import pickle

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from advertorch import attacks


"""
Loads (pretrained) pytorch model
"""
def load_model(path_to_ckpt=None):
  # Resnet for now
  model = models.resnet18(pretrained=True)
  return model

"""
Loads clean input data and its corresponding labels
"""
def load_clean_inputs():
  # CIFAR-10 for testing

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ]) 

  dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

  data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

  return data_loader


"""
Dumps perturbed inputs to pkl files  
"""
def dump_to_pkl(data, labels, type_of_attack=None, idx_offset = 0):

  # Keeping track of the index
  idx = idx_offset

  assert len(data) == len(labels), "data and labels should be same length: data:{}, labels:{}".format(data.shape[0], labels.shape[0])
    

  for img, label in zip(data, labels):
    file_path = os.path.join('x_adv', type_of_attack, str(label))
    if not os.path.exists(file_path):
      os.makedirs(file_path)

    # Generate pkl file in path <type_of_attack>/<label>/***.pkl
    file_name = "{}.pkl".format(os.path.join(file_path, str(idx)))
    with open(file_name, "wb") as f:
      pickle.dump(img, f)
    idx += 1

"""
Generates perturbed adversarial inputs
"""
def generate_perturbed_data(model, data_loader):
  # Print list of attacks we're going to generate
  # list_of_adv_attacks = ["GradientSignAttack", "PGDAttack"]
  list_of_adv_attacks = ["GradientSignAttack"]

  tick = time()

  model = model.to(device)
  model.eval()

  for attk in list_of_adv_attacks:
    print("Generating {} attack...".format(attk))

    attk_method = getattr(attacks, attk)

    adversary = attk_method(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.1, 
        clip_min=0.0, clip_max=1.0, targeted=False
        )

    cnt = 0
    for data, labels in data_loader:
      data = data.to(device)
      labels = labels.to(device)

      adv_untargeted = adversary.perturb(data, labels)
      print("Saving x_adv for {} ~ {}".format(cnt, cnt + len(data)))
      # dump_to_pkl(adv_untargeted.tolist(), labels.tolist(), type_of_attack=attk, idx_offset=cnt)
      dump_to_pkl(adv_untargeted.tolist(), labels.tolist(), type_of_attack=attk, idx_offset=cnt)
      cnt += len(data)

  time_elapsed = time() - tick
  print('Adversarial generation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

import argparse

if __name__ == "__main__":

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  print("==============================")
  print("Need to (hard-code) specify the following:")
  print("==============================")
  print("Path to model, default model is Resnet-18")
  print("Types of adversarial attacks, default attack is Gradient Sign Attack")
  print("Type of dataset, default is CIFAR-10")
  print("Path to write adversarial examples, default is './x_adv/<type_of_attack>/<image_id>.pkl'")

  print("Generating using: {}".format(device))


  # Test data
  data_loader = load_clean_inputs()
  model = load_model()
  generate_perturbed_data(model, data_loader)















