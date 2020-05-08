"""
Given a dataset of x_adv, `evaluate.py` runs it through the model and 
computes the accuarcy.

Specify which dataset was used as the base data
for example, if x_adv was generated from x where x is CIFAR10,
specify it as CIFAR10

Also, specify which model we are going to test it on.
This could be the same model you used to generate the adversarial examples to begin with
Or, it can be a different model that has been robustly trained using the same dataset (More likely)
"""

from os import path
import torch
import pickle
from PIL import Image
from torchvision.datasets import DatasetFolder
from utils import baseline_data_loader, load_sl_baseline_model
from generate_adv import load_model # Move to utils.py later
import torchvision.transforms as transforms
import numpy as np


# To fix Too many open files error
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def pkl_loader(path):
  with open(path, "rb") as f:
    _data = pickle.load(f)

  _data = np.array(_data).astype(np.float32)
  # img = Image.fromarray((_data * 255).astype(np.uint8))
  #img = Image.fromarray(_data)
  #return img
  return _data

def evaluation_data_loader(model_name, dataset, attk_type, requires_inv_transform=False):
  # Don't use normalization because pkl files are already saved normalized
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])
  path_to_data = path.join('x_adv', model_name, dataset, attk_type)
  eval_dataset = DatasetFolder(path_to_data, pkl_loader, extensions=('pkl'), transform=transform)
  eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=512, shuffle=False, num_workers=4)
  return eval_data_loader
  
"""
Evaluate the base model with 
dataset_type = "CIFAR10", "SVHN" 
"""
def evaluate_baseline(dataset_type):
  data_loader = baseline_data_loader("CIFAR10")
  return data_loader


def evaluate(model, data_loader):
  model = model.to(device)
  model.eval()

  total = 0
  correct = 0
  with torch.no_grad():
    for data, labels in data_loader:
      data = data.to(device)
      labels = labels.to(device)

      outputs = model(data)

      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  return total, correct 

if __name__ == "__main__":

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  print("Evaluating robustness using: {}".format(device))

  #model_name = "sl_baseline_cifar10"
  model_name = "ssl_vat_v11_svhn"

  # Specify what type of dataset we're going to use
  dataset = "svhn"

  # Evaluate using x_adv
  # data_loader = baseline_data_loader("CIFAR10")

  #model = load_sl_baseline_model('pretrained/sl_base_v11_cifar10', with_weights=True)
  model = load_sl_baseline_model("pretrained/{}".format(model_name), with_weights=True)
  model = model.net
  model.name = model_name

  data_loader = evaluation_data_loader(model_name, dataset, "GradientSignAttack")
  total, correct = evaluate(model, data_loader)
  print(total, correct)
  print("Accuracy for GradientSign Attacks: {}".format(correct/total))

  data_loader = evaluation_data_loader(model_name, dataset, "PGDAttack")
  total, correct = evaluate(model, data_loader)
  print(total, correct)
  print("Accuracy for PGD Attacks: {}".format(correct/total))

  data_loader = evaluation_data_loader(model_name, dataset, "LinfPGDAttack")
  total, correct = evaluate(model, data_loader)
  print(total, correct)
  print("Accuracy for LinfPGD Attacks: {}".format(correct/total))

  data_loader = evaluation_data_loader(model_name, dataset, "L2PGDAttack")
  total, correct = evaluate(model, data_loader)
  print(total, correct)
  print("Accuracy for L2PGD Attacks: {}".format(correct/total))










  
