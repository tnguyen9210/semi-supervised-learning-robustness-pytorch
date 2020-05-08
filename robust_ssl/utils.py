import json
import torch
from torchvision import transforms
from torchvision import datasets
from models.sl_baseline.deep_model import DeepModel

"""
Creates clean data loader (baseline)
"""
def baseline_data_loader(dataset_type, train_or_test=False):
  if dataset_type not in ["CIFAR10", "SVHN"]:
    raise Exception("dataset_type should be either CIFAR10 or SVHN")

  # CIFAR-10 for testing
  if dataset_type == "CIFAR10":
    normalize = transforms.Normalize(
      mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

  if dataset_type == "SVHN":
    normalize = transforms.Normalize(
      mean=[0.4377, 0.4438, 0.4728], std=[0.1975, 0.2004, 0.1964])

  transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
  ])

  dataset = getattr(datasets, dataset_type)

  if dataset_type == "CIFAR10":
    baseline_dataset = dataset(root='./data', train=train_or_test, download=True, transform=transform)
  elif dataset_type == "SVHN": 
    split_to_use = 'train' if train_or_test == True else 'test'
    baseline_dataset = dataset(root='./data', split=split_to_use, download=True, transform=transform)

  data_loader = torch.utils.data.DataLoader(baseline_dataset, batch_size=512, shuffle=False, num_workers=8)
  return data_loader

"""
Load model configuration
"""
def load_config(path_to_config, verbose=True):
  with open("{}/config.json".format(path_to_config)) as fin:
    config = json.load(fin)
  print("Config loaded from file {}".format(path_to_config))
  return config

"""
Load Supervised Learning baseline model
"""
def load_sl_baseline_model(model_dir, with_weights=False):
  model_args = load_config(model_dir)
  model = DeepModel(model_args)
  if with_weights:
    model.load_state("{}/best_model.ckpt".format(model_dir))
  return model

"""
Load model weights
"""
def load_model_weights(model, path_to_weights):
  pretrained_model = None
  return pretrained_model
