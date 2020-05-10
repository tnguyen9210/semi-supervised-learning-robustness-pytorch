import torch
from utils import load_sl_baseline_model, baseline_data_loader

if __name__ == "__main__":

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("Running on: {}".format(device))

  # Run performance of baseline models

  # load testset for both CIFAR10 and SVHN
  dl_cifar10 = baseline_data_loader('CIFAR10')
  dl_svhn = baseline_data_loader('SVHN')

  # First test for CIFAR model
  model_type = "baseline"
  dataset_type = "CIFAR10"

  model = load_sl_baseline_model('pretrained/sl_base_v11_cifar10', with_weights=True)
  net = model.net.to(device)
  net.eval()

  total = len(dl_cifar10.dataset)
  correct = 0.

  with torch.no_grad():
    for data, label in dl_cifar10:
      data = data.to(device)
      label = label.to(device)
      logit = net(data)
      prediction = torch.argmax(logit, dim=1)
      correct += torch.sum(prediction == label).item()

  print("Test complete ----")
  print("Model: {}, Dataset: {}".format(model_type, dataset_type))
  print("Test images: {}, Correct Images: {}".format(total, correct))
  print("Accuracy: {}".format(correct / total))

  # Second test for SVHN model and data
  model_type = "baseline"
  dataset_type = "SVHN"

  model = load_sl_baseline_model('pretrained/sl_base_v11_svhn', with_weights=True)
  net = model.net.to(device)
  net.eval()

  total = len(dl_svhn.dataset)
  correct = 0.

  with torch.no_grad():
    for data, label in dl_svhn:
      data = data.to(device)
      label = label.to(device)
      logit = net(data)
      prediction = torch.argmax(logit, dim=1)
      correct += torch.sum(prediction == label).item()

  print("Test complete ----")
  print("Model: {}, Dataset: {}".format(model_type, dataset_type))
  print("Test images: {}, Correct Images: {}".format(total, correct))
  print("Accuracy: {}".format(correct / total))
