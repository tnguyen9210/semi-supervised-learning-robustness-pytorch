
import logging
import operator


def print_config(config):
    info = "Running with the following configs:\n"
    for k,v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    # print("\n" + info + "\n")
    return info


def set_logger(log_path=None, verbose=True):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # Logging to a file
        if log_path:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

        # Logging to console
        if verbose:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)

    return logger

class EarlyStopper(object):
    def __init__(self, patience, mode, soft_eps, hard_thres):
        self.soft_patience = patience
        self.hard_patience = 25
        self.soft_cnt = 0
        self.hard_cnt = 0
        self.soft_eps = soft_eps
        self.hard_thres = hard_thres
        self.best_score = None
        if mode == "max":
            self.comp_func = operator.le
        elif mode == "min":
            self.comp_func = operator.ge

    def check(self, score, logger):
        # hard early stop when dev score has not reached desire thres
        if score < self.hard_thres:
            self.hard_cnt += 1
            if self.hard_cnt >= self.hard_patience:
                logger.info("hard early stop!")
                return True
            
        # soft early stop when dev score has not improved after n epochs
        if self.best_score is None:
            self.best_score = score
        elif self.comp_func(score, self.best_score):
            if self.comp_func(score + self.soft_eps, self.best_score):
                self.soft_cnt += 1
                if self.soft_cnt >= self.soft_patience:
                    logger.info("soft early stop!")
                    return True
        else:
            self.best_score = score
            self.soft_cnt = 0
            
        return False 
