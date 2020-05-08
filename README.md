# semi-supervised-learning-robustness-pytorch
Code for "Catching Generalization and Robustness with One Stone: A Unified Semi-supervised Learning Framework"

# Datasets

All preprocessed datasets for CIFAR10 and SVHN should be placed in dirs `data/cifar10_v11` and `data/svhn_v11` respectively.

# Models

sl_base_v11: full supervised learning with all labeled train dataset 

ssl_base_v11: supervised learning with few labeled train dataset

ssl_vat_v11: vat semi-supervised learning with few labeled dataset and unlabeled dataset

Notes: the outer version is the version of generated dataset (e.g., sl_base_v11), the inner version is the version of model's architecture (e.g., sl_base_v11_cifar10)

# How to train each model

```python
python train.py
```
All models are set with default parameters that can reproduce the performance results. If you want to modify parameters go to `config.py`.

# How to test attacker

All testing can be done in dir `10_test_adv_v11` with the following instructions:

1. copy the pretrained models into dir `10_test_adv_v11/saved_models`
2. run cmd `python --test_attacks.py --model_id=sl_base_v11_cifar10 --eval_set test_lbl`

where `--model_id` is the name of the model.
