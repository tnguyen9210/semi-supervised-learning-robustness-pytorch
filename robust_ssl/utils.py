import torch
from torchvision import transforms
from torchvision import datasets

"""
Creates clean data loader (baseline)
"""
def baseline_data_loader(dataset_type):
  if dataset_type not in ["CIFAR10", "SVHN"]:
    raise Exception("dataset_type should be either CIFAR10 or SVHN")

  # CIFAR-10 for testing

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  dataset = getattr(datasets, dataset_type)

  baseline_dataset = dataset(root='./data', train=False, download=True, transform=transform)
  data_loader = torch.utils.data.DataLoader(baseline_dataset, batch_size=64, shuffle=False, num_workers=8)

  return data_loader
