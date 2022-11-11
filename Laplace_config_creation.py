#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pprint
import os


# In[9]:


# This block creates ALL the config files!
# This block creates ALL the config files!

general_json = {
                'Metric': ["Laplace", "MC", "MLL", "AIC"],
                'Kernel_search': ["CKS"],
                'train_data_ratio': [0.5],#50% of the eval data
                'Data_kernel': ["SIN"],
                'weights': [[1., 1.]],
                'Variance_list': [4],
                'eval_START':[-10.0],
                'eval_END':[10.0],
                'eval_COUNT':[200],
                'optimizer':['Adam'],
                'train_iterations':[100],
                'LR': [0.1],
                'Noise': [0.0],#1%, 5%, 10% of max
                'Data_scaling': [False]
               }

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

# In[10]:


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


# In[ ]:





# In[11]:


import json
# Assumption: list is ordered to be correct with the keys
# Output: A dict with
def generate_config(keys, configurations):
    # keys is a list of 4 elements
    # entries 0 and 2 are the _main_ keys of the respective dictionaries
    # entries 1 and 3 are lists of the keys of the individual list entries (ordered!)
    # configurations is a list of tuples, where the first tuple element is part of the entry 0 configuration
    # and the second tuple element is part of entry 2 configuration

    # create a config file for each configuration and store it at 'configs/'
    for i, config in enumerate(configurations):
        config_dict = {}
        config_dict = {key:value for key, value in zip(keys, config[0])}
        #config_dict[keys[2]] = {key:value for key, value in zip(keys[3], config[1])}
        config_file_name = f"{config_dict['Metric']}_{i}"
        conf_dir = config_dict['Metric']
        with open(os.path.join("configs", f"{conf_dir}", f"{config_file_name}.json"), 'w') as configfile:
            configfile.write(json.dumps(config_dict, indent=4))


# In[12]:


general = json_iter_combinations(general_json)
keys = list(general_json.keys())
config = list(itertools.product(general))


# In[13]:


print(len(config))


# In[7]:


generate_config(keys, config)
