
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--data_dir', type=str, default="../../data/base_v11")
    parser.add_argument('--domain', type=str, default="svhn_trans")
    parser.add_argument('--img_size', type=int, default=32)

    # Training
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of total training epochs.")
    parser.add_argument('--batch_size', type=int, default=128, help="Training batch size.")

    # Optim
    parser.add_argument('--optim', type=str, default='sgd', help="Optimizer: sgd, adagrad, adam or adamax.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum.")
    parser.add_argument('--l2_params', type=float, default=0.005, help='L2 regularization for params.')
    parser.add_argument('--max_grad_norm', type=float, default=1, help="Max grad norm.")

    # VAT
    parser.add_argument('--vat_niters', type=int, default=1)
    parser.add_argument('--vat_eps', type=float, default=6.0)
    parser.add_argument('--vat_xi', type=float, default=0.01)
    
    # Feature encoder (CNNs)
    parser.add_argument('--enc_nlayers', type=int, default=2)
    parser.add_argument('--enc_kernel_size', type=int, default=5)
    parser.add_argument('--enc_num_channels1', type=int, default=64)
    parser.add_argument('--enc_num_channels2', type=int, default=128)
    parser.add_argument('--enc_num_channels3', type=int, default=128)
    parser.add_argument('--enc_droprate1', type=float, default=0.5)
    parser.add_argument('--enc_droprate2', type=float, default=0.5)
    parser.add_argument('--enc_droprate3', type=float, default=0.5)

    # Image classifier (FCs)
    parser.add_argument('--img_cls_nlayers', type=int, default=2)
    parser.add_argument('--img_cls_hidden_dim1', type=int, default=128)
    parser.add_argument('--img_cls_hidden_dim2', type=int, default=128)
    parser.add_argument('--img_cls_droprate1', type=float, default=0.5)
    parser.add_argument('--img_cls_droprate2', type=float, default=0.5)

    # Logging, Saving and Loading
    parser.add_argument('--model_id', type=str, default='0_0', help='ID under which to save models.')
    parser.add_argument('--ckpt_name', type=str, default='best_model.ckpt', help='Filename of the pretrained model.')
    parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
    
    
    return parser.parse_args()
