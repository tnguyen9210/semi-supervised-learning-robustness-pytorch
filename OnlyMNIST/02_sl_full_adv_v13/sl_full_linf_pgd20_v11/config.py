
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--data_dir', type=str, default="../../data/mnist_v13")
    parser.add_argument('--domain', type=str, default="mnist_orig")
    parser.add_argument('--img_size', type=int, default=28)

    # Training
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_iters_per_epoch', type=int, default=1000)
    
    # Optim
    parser.add_argument('--optim', type=str, default='sgd', help="Optimizer: sgd, adagrad, adam or adamax.")
    parser.add_argument('--lr', type=float, default=0.05, help="Learning rate.")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum.")
    parser.add_argument('--l2_params', type=float, default=5e-4, help='L2 regularization for params.')
    parser.add_argument('--max_grad_norm', type=float, default=1, help="Max grad norm.")
    parser.add_argument('--scheduler_t0', type=int, default=10)
    parser.add_argument('--scheduler_tmult', type=int, default=1)

    # Feature encoder (CNNs)
    parser.add_argument('--lenet_depth', type=int, default=28)
    parser.add_argument('--lenet_widen_factor', type=int, default=1)
    parser.add_argument('--lenet_droprate', type=float, default=0.3)

    # Image classifier (FCs)
    parser.add_argument('--fc_nlayers', type=int, default=1)
    parser.add_argument('--fc_hidden_dim', type=int, default=1024)
    parser.add_argument('--fc_droprate', type=float, default=0.3)

    # Logging, Saving and Loading
    parser.add_argument('--model_id', type=str, default='10', help='ID under which to save models.')
    parser.add_argument('--model_dir', type=str, default='./saved_models')
    parser.add_argument('--eval_set', type=str, default='test_lbl')
    parser.add_argument('--ckpt_name', type=str, default='best_model.ckpt', help='Filename of the pretrained model.')
    parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
    
    return parser.parse_args()
