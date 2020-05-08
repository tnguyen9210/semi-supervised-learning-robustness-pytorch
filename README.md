# Semi-supervised Learning Robustness Pytorch
Code for "Catching Generalization and Robustness with One Stone: A Unified Semi-supervised Learning Framework"

## Datasets

We use the following dataset formats for both CIFAR10 and SVHN domains

- `domain_test_lbl`: test dataset 

- `domain_dev_lbl`: dev dataset, splitted from original train dataset 

- `domain_train_lbl`: train dataset, original train dataset minus dev dataset 

- `domain_train_lbl_numlbl`: labeled train dataset, splitted from train_lbl

- `domain_train_unlbl_numlbl`: unlabeled train dataset, train_lbl dataset minus lbl_num_lbl

For pre-processing, CIFAR10 datasets are applied global contrast normalization and ZCA whitening. SVHN datasets are applied global contrast normalization. These pre-processing steps are needed for the performance and needed for our `ImageDataset` class (instead of generic torchvision datasets).

All preprocessed datasets for CIFAR10 and SVHN should be placed in dirs `data/cifar10_v11` and `data/svhn_v11` respectively.

## Models

We use the following models for both CIFAR10 and SVHN domains

- `sl_base`: full supervised learning with all labeled train dataset 

- `ssl_base`: supervised learning with few labeled train dataset

- `ssl_vat`: vat semi-supervised learning with few labeled dataset and unlabeled dataset

Notes: the outer version is the version of generated dataset (e.g., sl_base_v11), the inner version is the version of model's architecture (e.g., sl_base_v11_cifar10)

## How to train each model

Run cmd `python train.py`

All models are set with default parameters that can reproduce the performance results. If you want to modify parameters go to `config.py`.

## How to test attacker

All testing can be done in dir `10_test_adv_v11` with the following instructions:

1. Place the pretrained models into dir `10_test_adv_v11/saved_models`
2. Run cmd `python --test_attacks.py --model_id=sl_base_v11_cifar10 --eval_set test_lbl`

where `--model_id` is the name of the model.
