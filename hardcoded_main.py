import copy
import configparser
from experiment_functions import Experiment
from GaussianProcess import ExactGPModel
from globalParams import options
import gpytorch
from helpFunctions import get_string_representation_of_kernel as gsr, clean_kernel_expression
from helpFunctions import amount_of_base_kernels
import json
from kernelSearch import *
from matplotlib import pyplot as plt
from metrics import *
from multiprocessing import Pool
import numpy as np
import os
import pdb
import random
import tikzplotlib
import time
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_text="RBF", weights=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel_text == "RBF":
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel_text == "SIN":
            self.covar_module = gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "SIN+RBF":
            if weights is None:
                self.covar_module = gpytorch.kernels.PeriodicKernel() + gpytorch.kernels.RBFKernel()
            else:
                self.covar_module = weights[0]*gpytorch.kernels.PeriodicKernel() + weights[1]*gpytorch.kernels.RBFKernel()
        elif kernel_text == "SIN*RBF":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.RBFKernel()
        elif kernel_text == "SIN*LIN":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.LinearKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



def load_config(config_file):
    with open(os.path.join("configs", config_file), "r") as configfile:
        temp = configfile.readlines()
        text ="".join(i.replace("\n","") for i in temp)
    var_dict = json.loads(text)

    # Create an experiment name based on the experiment setup
    # e.g. "CKS_Laplace_LR-0.1_SIN-RBF" etc.
    #name_vars = ["Metric", "Kernel_search", "Data_kernel", "LR", "Noise"]
    name_vars = [var_dict[key] for key in var_dict]
    experiment_name = "_".join([str(key) for key in name_vars])
    var_dict["experiment name"] = experiment_name

    return var_dict







def run_experiment(config):
    """
    This contains the training, kernel search, evaluation, logging, plotting.
    It takes an input file, processes the whole training, evaluation and log
    Returns nothing

    """

    eval_START = -5
    eval_END = 5
    eval_COUNT = 100
    optimizer = "Adam"
    train_iterations = 200
    LR = 0.1
    #noise =
    data_scaling = var_dict["Data_scaling"]
    use_BFGS = True
    num_draws = 1000 if metric == "MC" else None
    parameter_punishment = 2.0 if metric == "Laplace" else None

    # set training iterations to the correct config
    options["training"]["max_iter"] = int(train_iterations)

    # Data generation process
    observations_x = torch.linspace(eval_START, eval_END, eval_COUNT)
    if data_gen == "PER":
        observations_y = torch.sin(observations_x * 2*np.pi)
    elif data_gen == "4PER":
        observations_y = torch.sin(observations_x * 2*np.pi) + 0.5*torch.sin(observations_x * 3*np.pi) + 0.2*torch.sin(observations_x * 7*np.pi) + 0.1*torch.sin(observations_x * 11*np.pi)
    elif data_gen == "LIN*PER":
        observations_y = torch.sin(observations_x * 2*np.pi) * observations_x


    EXPERIMENT_REPITITIONS = 1
    for exp_num in range(EXPERIMENT_REPITITIONS):
        log_name = "..."
        experiment_keyword = var_dict["experiment name"]
        experiment_path = os.path.join("results", metric, f"{experiment_keyword}")
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        log_experiment_path = os.path.join(experiment_path, f"{experiment_keyword}")
        experiment = Experiment(log_experiment_path, exp_num, attributes=var_dict)

        # Initialize the model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        list_of_variances = [float(variance_list_variance) for i in range(28)]


        # Perform a training for AIC and Laplace


        # Perform MCMC











if __name__ == "__hardcoded_main__":
    with open("FINISHED.log", "r") as f:
        finished_configs = [line.strip().split("/")[-1] for line in f.readlines()]
    curdir = os.getcwd()
    keywords = ["PER", "4PER", "LIN*PER"]
    for config in keywords:
        run_experiment(config)

