
import json, ast
from pprint import pprint
from collections import defaultdict, OrderedDict

import numpy as np
import numpy.random as rnd

from train import train
from search_hyperparams import fixed_params, search_params

def main():
    # load results
    with open("logger/search_results.json", 'r') as fi:
        results = json.load(fi)

    # get best results
    sorted_results = \
        sorted([items for items in results.items() if items[1] != None],
               key=lambda items: items[1][1], reverse=True)
    best_configs = sorted_results[:10]
    pprint(best_configs)

    
    
def print_param_stats(results, score_thres):
    for param in search_params:
        print(f"\n--> {param}")
        param_scores = defaultdict(list)
        for config, scores in results.items():
            config = dict(ast.literal_eval(config))
            if (scores is not None) and (scores[0] >= score_thres):
                param_scores[config[param]].append(scores[0])

        param_stat = {param: [round(np.mean(scores), 4),
                              round(np.max(scores), 4)]
                      for param, scores in param_scores.items()}
        param_stat = sorted(
            param_stat.items(), key=lambda kv: kv[1][0], reverse=True)
        pprint(param_stat, width=40)


if __name__ == '__main__':
    main()

