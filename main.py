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
# ~~11. Loss thresholds (-1, -2, -3)?~~
# 12. Runtime evaluations
# 13. Calculate Fourier series of a signal and cut after n to generate data
# 14. Vary the values in the variance_list for the different parameters
#     (broader and narrower)

# Not goals of experiments
# - Parameter training



# Things to do in config
# 14. Vary the values in the variance_list for the different parameters
# 10. Training runtimes (100, 200, 300 iterations)?
# 7. How high is the threshold to detect structures? (0.9*sin + 0.1*RBF)
# 5. Test out noise levels and impacts

# Things to do in code
# 13. Calculate Fourier series of a signal/function and cut after n for data ?
#     Purpose: Gives a linear combination of sine/cosine and "represent" real
#     data
# 12. Runtime evaluations (Logging)
# 3. Reproducibility experiments (Logging/Repititions)

# Things to do afterwards
# 1. Check if Laplace Approx is better than MLL/MCMC
# 12. Runtime evaluations



# Things to log
# - Parameters and their values (post training)
# - The kernels that were tried (basically a progress tree)
# - Loss at each kernel try
# - The metrics with those kernels
#   - Question: Do we calculate ALL metrics and highlight the current choice?
# - Finale kernel
# - Parameter starting points at random restarts?
#   - Q: What would/could we hope to learn from this?



# Known issues:
# - The periodic kernel is prone to creating a non-PSD matrix
# - The MC metric is prone to creating non-PSD matrices due to the random parametrizations


import torch
import gpytorch
from matplotlib import pyplot as plt
import pdb
import random
import configparser
import os
import time
from multiprocessing import Pool
import tikzplotlib
from experiment_functions import Experiment
import copy
import json

from GaussianProcess import ExactGPModel
from helpFunctions import get_string_representation_of_kernel as gsr, clean_kernel_expression
from helpFunctions import amount_of_base_kernels
from kernelSearch import *


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_text="RBF"):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel_text == "RBF":
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel_text == "SIN":
            self.covar_module = gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "SIN+RBF":
            self.covar_module = gpytorch.kernels.PeriodicKernel() + gpytorch.kernels.RBFKernel()
        elif kernel_text == "SIN*RBF":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.RBFKernel()
        elif kernel_text == "SIN*LIN":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.LinearKernel()



    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



def RMSE(goal, prediction):
    mse = np.mean((goal-prediction)**2)
    rmse = np.sqrt(mse)
    return mse


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



def log(message, logfile="default.log"):
    with open(os.path.join(path, filename), 'w') as temp_file:
        temp_file.write(message)
    return 0




def run_experiment(config_file):
    """
    This contains the training, kernel search, evaluation, logging, plotting.
    It takes an input file, processes the whole training, evaluation and log
    Returns nothing

    """

    EXPERIMENT_REPITITIONS = 1

    ### Initialization
    var_dict = load_config(config_file)

    train_data_ratio = var_dict["train_data_ratio"]
    eval_START = var_dict["eval_START"]
    eval_END = var_dict["eval_END"]
    eval_COUNT = var_dict["eval_COUNT"]
    optimizer = var_dict["optimizer"]
    metric = var_dict["Metric"]
    kernel_search = var_dict["Kernel_search"]
    data_kernel = var_dict["Data_kernel"]
    variance_list_variance = var_dict["Variance_list"]
    train_iterations = var_dict["train_iterations"]
    LR = var_dict["LR"]
    noise = var_dict["Noise"]
    data_scaling = var_dict["Data_scaling"]



    log_name = "..."
    experiment_keyword = var_dict["experiment name"]
    experiment_path = os.path.join("results", f"{experiment_keyword}")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    log_experiment_path = os.path.join(experiment_path, f"{experiment_keyword}.log")
    experiment = Experiment(log_experiment_path, EXPERIMENT_REPITITIONS)

    for key in var_dict:
        experiment.store_result(key, var_dict[key])


        ## Create train data
    # Create base model to generate data

    # training data for model initialization (e.g. 1 point with x=0, y=0) ; this makes initializing the model easier
    prior_x = torch.linspace(0,1,1)
    prior_y = prior_x
    # initialize likelihood and model
    data_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    data_model = ExactGPModel(prior_x, prior_y, data_likelihood, kernel_text=data_kernel)
    observations_x = torch.linspace(eval_START, eval_END, eval_COUNT)
    # Get into evaluation (predictive posterior) mode
    data_model.eval()
    data_likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.prior_mode(True):
        observed_pred_prior = data_likelihood(data_model(observations_x))
        f_preds = data_model(observations_x)
        mean_prior = observed_pred_prior.mean
        lower_prior, upper_prior = observed_pred_prior.confidence_region()

    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_covar = f_preds.covariance_matrix
    observations_y = f_preds.sample()           # samples from the model


    X = observations_x[int((1-train_data_ratio)*0.5*eval_COUNT):int((1+train_data_ratio)*0.5*eval_COUNT)]
    Y = observations_y[int((1-train_data_ratio)*0.5*eval_COUNT):int((1+train_data_ratio)*0.5*eval_COUNT)]

    # Run CKS
    list_of_kernels = [gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())]
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    list_of_variances = [float(variance_list_variance) for i in range(28)]
    model, likelihood = CKS(X, Y, likelihood, list_of_kernels, list_of_variances, experiment, iterations=3, metric=metric)




    ### Model selection & Training (Jan)
    # Store the loss/Laplace/... at each iteration of the model selection!
    # Store the selected kernels over time in a parseable way
    # Store the parameters over time
    #
    #monte_carlo_simulate(model, likelihood, 100)

    ### Calculating various metrics
    # Metrics
    # - RMSE?
    #
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.prior_mode(True):
        observed_pred_prior = likelihood(model(observations_x))
        f_preds = model(observations_x)
        mean_posterior = observed_pred_prior.mean
        lower_posterior, upper_posterior = observed_pred_prior.confidence_region()

    mean_y = model(observations_x).mean

    f, ax = plt.subplots()
    f, ax = model.plot_model(return_figure=True, figure = f, ax=ax)
    ax.plot(observations_x, observations_y, 'k*')
    # Store the plots as .png

    # Store the plots as .tex



    eval_rmse = RMSE(observations_y.numpy(), mean_y.detach().numpy())
    print(eval_rmse)
    experiment.store_result("eval RMSE", eval_rmse)

    ### Plotting

    experiment.write_results()
    return 0





if __name__ == "__main__":
    with open("FINISHED.log", "r") as f:
        finished_configs = [line.strip() for line in f.readlines()]
    curdir = os.getcwd()
    KEYWORD = "MCMC"
    configs = os.listdir(os.path.join(curdir, "configs", KEYWORD))
    # Check if the config file is already finished and if it even exists
    configs = [os.path.join(KEYWORD, c) for c in configs if not os.path.join(KEYWORD, c) in finished_configs and os.path.isfile(os.path.join(curdir, "configs", KEYWORD, c))]
    for config in configs:
        run_experiment(config)
    #with Pool(processes=7) as pool:
    #    pool.map(run_experiment, configs)
