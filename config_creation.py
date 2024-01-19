#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pprint
import os
import pdb

import copy

def make_hash(o):

  """
  Makes a hash from a dictionary, list, tuple or set to any level, that contains
  only other hashable types (including any lists, tuples, sets, and
  dictionaries).
  """

  if isinstance(o, (set, tuple, list)):

    return tuple([make_hash(e) for e in o])    

  elif not isinstance(o, dict):

    return hash(o)

  new_o = copy.deepcopy(o)
  for k, v in new_o.items():
    new_o[k] = make_hash(v)

  return hash(tuple(frozenset(sorted(new_o.items()))))

# In[9]:



# This block creates ALL the config files!

general_json = {
    'Kernel_search': ["CKS"],
    'train_data_ratio': [0.5],#50% of the eval data
    #'Data_kernel': ["SE", "PER", "MAT32", "PER*SE", "PER+SE", "MAT32*PER", "MAT32+PER", "MAT32+SE", "MAT32*SE"],
    #'Data_kernel': ["PER", "SE", "MAT32", "MAT32+SE", "MAT32*SE", "MAT32*PER", "MAT32+PER", "PER*SE"],
    'Data_kernel': ["SE", "SE+SE", "LIN", "MAT32"],
    'weights': [[1., 1.]],
    'Variance_list': [4],
    'eval_START':[-5.0],
    'eval_END':[5.0],
    'eval_COUNT':[5, 10, 20, 30, 40, 50, 100, 200],#, 250, 500],
    'optimizer':['PyGRANSO'],
    'train_iterations':[100],
    'LR': [0.1],
    'Noise': [0.0],#1%, 5%, 10% of max
    'Data_scaling': [True],
    'BFGS' : [False]
}

MC_json = {
    "Metric": ["MC"],
    "num_draws": [1000]
}

Laplace_json = {
    "Metric": ["Laplace"],
    "parameter_punishment": [0.0, -1.0, "BIC"]
}

#Laplace_prior_json = {
#    "Metric": ["Laplace_prior"],
#    "parameter_punishment": [0.0, -1.0, "BIC"]
#
MAP_json = {
    "Metric": ["MAP"]
}

MLL_json = {
    "Metric": ["MLL"]
}

AIC_json = {
    "Metric" : ["AIC"]
}

BIC_json = {
    "Metric" : ["BIC"]
}

specific_jsons = [MC_json, MLL_json, MAP_json,  AIC_json, BIC_json, Laplace_json] # , Laplace_json
#general_json = {
#                'Metric': ["Laplace", "MC", "MLL", "AIC"],
#                'Kernel_search': ["CKS"],
#                'train_data_ratio': [0.5],#50% of the eval data
#                'Data_kernel': ["SIN", "RBF", "SIN*RBF", "SIN+RBF"],
#                'weights': [[0.9, 0.1], [0.7, 0.3], [1., 1.], [0.3, 0.7], [0.1, 0.9]],
#                'Variance_list': [1, 4, 8, 16],
#                'eval_START':[-10.0],
#                'eval_END':[10.0],
#                'eval_COUNT':[200],
#                'optimizer':['Adam'],
#                'train_iterations':[100, 200, 300],
#                'LR': [0.1],
#                'Noise': [0.0, 0.01, 0.05, 0.1],#1%, 5%, 10% of max
#                'Data_scaling': [False]
#               }


# Expects dict of mixture of lists and other data
from collections.abc import Iterable
import itertools

def json_iter_combinations(json_iterable):
    keys = list(json_iterable.keys())
    key = keys.pop(0)
    if not isinstance(json_iterable[key], Iterable):
        data1 = [json_iterable[key]]
    else:
        data1 = json_iterable[key]
    total_combinations = list()
    if keys == []:
        return [data1]
    for key in keys:
        if not isinstance(json_iterable[key], Iterable):
            new_data = [json_iterable[key]]
        else:
            new_data = json_iterable[key]
        if total_combinations == []:
            total_combinations = list(itertools.product(data1, new_data))
        else:
            total_combinations = list(itertools.product(total_combinations, new_data))
            total_combinations = [[*entry[0], entry[1]] for entry in total_combinations]
    return total_combinations



import json
# Assumption: list is ordered to be correct with the keys
# Output: A dict with
def generate_config(keys, configurations):
    # keys is a list of 4 elements
    # entries 0 and 2 are the _main_ keys of the respective dictionaries
    # entries 1 and 3 are lists of the keys of the individual list entries (ordered!)
    # configurations is a list of tuples, where the first tuple element is part of the entry 0 configuration
    # and the second tuple element is part of entry 2 configuration

    try:
    # create a config file for each configuration and store it at 'configs/'
        for i, config in enumerate(configurations):
            config_dict = {}
            config_dict = {key:value for key, value in zip(keys, config)}
            #config_dict[keys[2]] = {key:value for key, value in zip(keys[3], config[1])}
            config_file_name = "_".join([str(config_dict[k]) for k in config_dict]) 
            #config_file_name = f"{config_dict['Metric']}_{i}"
            conf_dir = config_dict['Metric']

            if not os.path.exists(os.path.join("configs", conf_dir)):
                os.makedirs(os.path.join("configs", conf_dir))
            with open(os.path.join("configs", f"{conf_dir}", f"{config_file_name}.json"), 'w') as configfile:
                configfile.write(json.dumps(config_dict, indent=4))
    except Exception as e:
        pdb.post_mortem()


general = json_iter_combinations(general_json)
keys = list(general_json.keys())

num_confs = 0
for task_specific_json in specific_jsons:
    task_spec = json_iter_combinations(task_specific_json)
    keys = list(general_json.keys())
    keys.extend(list(task_specific_json.keys()))
    config = [[*a, *b] for a, b in itertools.product(general, task_spec)]
    num_confs += len(config)
    generate_config(keys, config)

print(num_confs)