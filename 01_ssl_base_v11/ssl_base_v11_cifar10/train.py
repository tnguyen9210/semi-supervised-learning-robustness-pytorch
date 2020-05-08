
from collections import defaultdict, OrderedDict
import numpy as np

import torch

from core.deep_model import DeepModel

from dataset import load_data
from config import parse_args
from utils import print_config, save_config, set_logger, EarlyStopper


def train(config=None):
    torch.backends.cudnn.benchmark = True  # speed up cudnn
    
    args = parse_args()
    args = vars(args)
    if config:
        args.update(config)
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    
    # set logger
    model_dir = f"{args['model_dir']}/{args['model_id']}"
    logger = set_logger(f"logger/{args['model_id']}", verbose=True)
    logger.info(print_config(args))
    save_config(args, model_dir, verbose=True)
    
    # load data
    train_lbl, dev_lbl, test_lbl = \
        load_data(args['domain'], args['data_dir'], args['img_size'],
                  args['num_iters_per_epoch'], args['batch_size'])

    eval_datasets = OrderedDict(
        {'dev': dev_lbl, 'test': test_lbl})
    eval_perfs = defaultdict()
    for dn in eval_datasets:
        eval_perfs[dn] = defaultdict()
        
    # set model
    model = DeepModel(args)

    # create early stopper
    early_stopper = EarlyStopper(
        patience=100, mode="max", soft_eps=0.2, hard_thres=85)
    
    # train model 
    best_score = 1000.
    best_epoch = 1
    for epoch in range(args['num_epochs']):
        logger.info(">>>>>>>>>>>>>>TRAINING<<<<<<<<<<<<<<<<<<<<")
        train_loss = model.update(train_lbl, epoch, logger)

        logger.info(">>>>>>>>>>>>>>EVALUATING<<<<<<<<<<<<<<<<<<")
        for ds in eval_datasets:
            eval_perfs[ds][epoch] = model.evaluate(eval_datasets[ds])
            print_perfs(ds, eval_perfs[ds][epoch], logger)

        # select best model instance that has mimimum train loss
        score = eval_perfs['dev'][epoch]['error']
        if score < best_score:  
            best_score = score
            best_epoch = epoch
            logger.info(">>>>>>>>>>>>>>NEW BEST PERFORMANCE<<<<<<<<")
            # save the best state
            model.save_state(model_dir)

        # execute early stop if dev score does not improve after n epochs
        # dev_score = eval_perfs['dev'][epoch]['acc']
        # if early_stopper.check(dev_score, logger):
        #     break

    logger.info(f">>>>>>>>>>>>>>BEST PERFORMANCE, epoch {best_epoch}<<<<<<<<<<<<<<<<<<")
    for ds in eval_datasets:
        print_perfs(ds, eval_perfs[ds][best_epoch], logger)

    logger.handlers = []
    return [eval_perfs[ds][best_epoch]['acc'] for ds in eval_datasets]


def print_perfs(ds, perfs, logger):
    logger.info(f"{ds} : error = {perfs['error']}")


if __name__ == '__main__':
    train()
