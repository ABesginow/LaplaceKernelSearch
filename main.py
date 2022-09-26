# Purpose of the script
# Automatic run of experiments with configurations taken from a config file/dir
# Goals of experiments:
# 1. Check if Laplace Approx is better than MLL/MCMC
# 2. Real data experiments
# 3. Reproducibility experiments
# 4. "Better" structure discovery?
# 5. Test out noise levels and impacts
# 6. Vary the kernel algorithms?
# 7. How high is the threshold to detect structures? (0.9*sin + 0.1*RBF)
# 8. Ablation: Physical kernel ?
# 9. Ablation: Robust against different parametrizations?
# 10. Training runtimes (100, 200, 300 iterations)?
# 11. Loss thresholds (-1, -2, -3)?

# Not goals of experiments
# - Parameter training





import torch
import gpytorch
from matplotlib import pyplot as plt
import pdb
import random
import configparser
import time
from multiprocessing import Pool






def load_config(config_file):
    with open(os.path.join("configs", config_file), "r") as configfile:
        text = configfile.readline()
    config = json.loads(text)

    # load variables like learning rate etc.


    # Create an experiment name based on the experiment setup
    # e.g. "CKS_Laplace_LR-0.1_SIN-RBF" etc.
    experiment_name = ...

    return var_dict


def log(logfile="default.log"):
    return 0







def run_experiment(config_file):
    """
    This contains the training, kernel search, evaluation, logging, plotting.


    """

    return 0





if __name__ == "__main__":
    with open("FINISHED.log", "r") as f:
        finished_configs = [line.strip() for line in f.readlines()]
    curdir = os.getcwd()
    KEYWORD = "something"
    configs = os.listdir(os.path.join(curdir, "configs", KEYWORD))
    # Check if the config file is already finished and if it even exists
    configs = [os.path.join(KEYWORD, c) for c in configs if not os.path.join(KEYWORD, c) in finished_configs and os.path.isfile(os.path.join(curdir, "configs", KEYWORD, c))]


    for config in configs:
        run_experiment(config)
    #with Pool(processes=7) as pool:
    #    pool.map(run_experiment, configs)
