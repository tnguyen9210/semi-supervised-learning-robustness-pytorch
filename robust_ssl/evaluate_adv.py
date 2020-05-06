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

import os 
import torch
from torchvision.datasets import DatasetFolder
from utils import baseline_data_loader
from generate_adv import load_model # Move to utils.py later
from ssl_vat.ssl_vat_v23_svhn import utils


def pkl_loader(path):
  with open(path, "rb") as f:
    _data = pickle.load(f)
  return _data

def evaluation_data_loader(attk_type, requires_inv_transform=False):
  path_to_data = os.join('x_adv', attk_type)
  eval_dataset = DatasetFolder(path_to_data, pkl_loader, extensions=('pkl'))
  eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4)
  return eval_data_loader
  
"""
Evaluate the base model with 
dataset_type = "CIFAR10", "SVHN" 
"""
def evaluate_baseline(dataset_type):
  data_loader = baseline_data_loader("CIFAR10")


def evaluate(model, data_loader):
  total = 0
  correct = 0
  with torch.no_grad():
    for data, labels in data_loader:
      data.to(device)
      labels.to(device)

      outputs = model(data)

      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  return total, correct 

if __name__ == "__main__":

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  print("Evaluating robustness using: {}".format(device))
  load_config()

  # Test data
  data_loader = baseline_data_loader("CIFAR10")


  model = load_model()
  total, correct = evaluate(model, data_loader)

  print(total, correct)






