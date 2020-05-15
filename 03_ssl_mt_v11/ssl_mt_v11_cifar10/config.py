
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--data_dir', type=str, default="../../data/cifar10_v11")
    parser.add_argument('--domain', type=str, default="cifar10_orig")
    parser.add_argument('--img_size', type=int, default=32)

    # Training
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--num_iters', type=int, default=500000)
    parser.add_argument('--num_iters_per_epoch', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--consis_warmup', type=int, default=200000)
    
    # VAT
    parser.add_argument('--vat_lr', type=float, default=0.03)
    parser.add_argument('--vat_niters', type=int, default=1)
    parser.add_argument('--vat_eps', type=float, default=5.0)
    parser.add_argument('--vat_xi', type=float, default=1e-6)
    parser.add_argument('--vat_consis_coef', type=float, default=0.3)

    # Mean teacher (MT)
    parser.add_argument('--mt_lr', type=float, default=0.0004)
    parser.add_argument('--mt_ema_factor', type=float, default=0.95)
    parser.add_argument('--mt_consis_coef', type=float, default=8.0)

    # Optim
    parser.add_argument('--optim', type=str, default='sgd', help="Optimizer: sgd, adagrad, adam or adamax.")
    parser.add_argument('--lr', type=float, default=0.05, help="Learning rate.")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum.")
    parser.add_argument('--l2_params', type=float, default=5e-4, help='L2 regularization for params.')
    parser.add_argument('--max_grad_norm', type=float, default=1, help="Max grad norm.")
    parser.add_argument('--scheduler_t0', type=int, default=10)
    parser.add_argument('--scheduler_tmult', type=int, default=2)

    
    # Feature encoder (CNNs)
    parser.add_argument('--resnet_depth', type=int, default=28)
    parser.add_argument('--resnet_widen_factor', type=int, default=2)
    parser.add_argument('--resnet_group1_droprate', type=float, default=0.3)
    parser.add_argument('--resnet_group2_droprate', type=float, default=0.3)
    parser.add_argument('--resnet_group3_droprate', type=float, default=0.3)

    # Image classifier (FCs)
    parser.add_argument('--img_cls_nlayers', type=int, default=2)
    parser.add_argument('--img_cls_hidden_dim1', type=int, default=128)
    parser.add_argument('--img_cls_hidden_dim2', type=int, default=128)
    parser.add_argument('--img_cls_droprate1', type=float, default=0.5)
    parser.add_argument('--img_cls_droprate2', type=float, default=0.5)

    # Logging, Saving and Loading
    parser.add_argument('--model_id', type=str, default='10', help='ID under which to save models.')
    parser.add_argument('--model_dir', type=str, default='./saved_models')
    parser.add_argument('--eval_set', type=str, default='test_lbl')
    parser.add_argument('--ckpt_name', type=str, default='best_model.ckpt', help='Filename of the pretrained model.')
    parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
    
    return parser.parse_args()
