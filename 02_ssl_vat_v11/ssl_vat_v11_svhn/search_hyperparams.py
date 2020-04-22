""" 
Desc: Apply evolution algo to search for best hyperparams
"""

import json
from pprint import pprint, pformat
from collections import OrderedDict
import numpy.random as rnd

from train import train

seed = 0
fixed_params = {
    'seed': seed,
    'device': 'cuda:0',
    'num_epochs': 100,
    'batch_size': 128,
    'optim': 'adam',
    'l2_params': 1e-5,
    'enc_num_channels1': 64,
    'enc_num_channels2': 128,
    'enc_num_channels3': 128,
    'img_cls_hidden_dim1': 128,
}

search_params = {
    'lr': [0.0001, 0.0005, 0.001, 0.004],
    
    'enc_droprate1': [0.0, 0.2, 0.3, 0.4, 0.5, 0.6],
    'enc_droprate2': [0.0, 0.2, 0.3, 0.4, 0.5, 0.6],
    
    'img_cls_droprate1': [0.0, 0.2, 0.3, 0.4, 0.5, 0.6],

    'vat_niters': [1, 2],
    'vat_eps': [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
    'vat_xi': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
}

def main():
    rnd.seed(0)
    
    pop_size = 8                # population size
    elite_size = 4              # num of elite instances to be retained
    num_gens = 10               # number of generations
    mutate_rate = 0.7

    all_perfs = OrderedDict()
    best_perfs = OrderedDict()

    # init population perfs
    pop_perfs = OrderedDict()
    cnt = 0
    while cnt < pop_size:
        config = dict()
        for key, choices in search_params.items():
            val = choices[rnd.randint(0, len(choices))]
            config[key] = val
            
        # add config into all_perfs 
        config = tuple(sorted(config.items()))  # note: we can use tuple as hash key
        if str(config) not in all_perfs:
            pop_perfs[config] = None
            all_perfs[str(config)] = None  # but we can only save dict with str hash key
            cnt += 1

    # evolution algo
    gen_idx = -1
    model_id = 0 
    while gen_idx < num_gens:
        gen_idx += 1

        # reduce mutate, we explore less and be more greedy after 3rd generation
        if gen_idx > 3:         
            mutate_rate = 0.25

        # compute current population perfs
        for config in pop_perfs:
            model_id += 1
            config_dict = dict(config)
            config_dict.update(fixed_params)
            config_dict['model_id'] = f"{seed}_{model_id:03d}"
            try:
                score = train(config_dict)
                # score = [round(rnd.uniform(), 3) for _ in range(2)]
                score.append(model_id)
            except KeyboardInterrupt:
                score = [0, 0, model_id]
                input('press any key')
            pop_perfs[config] = score
            all_perfs[str(config)] = score

        # rank and select top population perfs
        best_perfs.update(pop_perfs)
        sorted_best_perfs = \
            sorted(best_perfs.items(), key=lambda x: x[1][0], reverse=True)
        best_perfs = OrderedDict(sorted_best_perfs[:elite_size])
        best_configs = list(best_perfs.keys())
                   
        # generate new population
        pop_perfs = OrderedDict()
        cnt = 0
        while cnt < pop_size:
            config = dict()
            
            # pick two random parents
            p0 = dict(best_configs[rnd.randint(0, elite_size)])
            p1 = dict(best_configs[rnd.randint(0, elite_size)])

            for key, choices in search_params.items():
                # crossover
                config[key] = p0[key] if rnd.random() < 0.5 else p1[key]

                # mutute
                if rnd.random() < mutate_rate:
                    while True:
                        val = choices[rnd.randint(0, len(choices))]
                        if val != config[key]:
                            break
                    config[key] = val

            # add config to all_perfs 
            config = tuple(sorted(config.items()))
            if str(config) not in all_perfs:
                pop_perfs[config] = None
                all_perfs[str(config)] = None
                cnt += 1
                
        if gen_idx % 1 == 0:
            with open("logger/search_results.json", 'w') as fo:
                json.dump(all_perfs, fo, indent=2)
                
    with open("logger/search_results.json", 'w') as fo:
        json.dump(all_perfs, fo, indent=2)


        
if __name__ == '__main__':
    main()
