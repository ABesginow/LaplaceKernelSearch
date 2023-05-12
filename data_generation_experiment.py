
import copy
import configparser
from experiment_functions import Experiment
from GaussianProcess import ExactGPModel
from globalParams import options
import gpytorch
from helpFunctions import get_string_representation_of_kernel as gsr, clean_kernel_expression
from helpFunctions import amount_of_base_kernels
from itertools import product
import json
from kernelSearch import *
from matplotlib import pyplot as plt
from multiprocessing import Pool
import os
import pdb
import pickle
import random
import tikzplotlib
import time
import torch


def get_kernels_in_kernel_expression(kernel_expression):
    """
    returns list of all base kernels in a kernel expression
    """
    if kernel_expression == None:
        return []
    if hasattr(kernel_expression, "kernels"):
        ret = list()
        for kernel in kernel_expression.kernels:
            ret.extend(get_kernels_in_kernel_expression(kernel))
        return ret
    elif kernel_expression._get_name() == "ScaleKernel":
        return get_kernels_in_kernel_expression(kernel_expression.base_kernel)
    elif kernel_expression._get_name() == "GridKernel":
        return get_kernels_in_kernel_expression(kernel_expression.base_kernel)
    else:
        return [kernel_expression]



def reparameterize(kernels):
    limits = {"RBFKernel": {"lengthscale": [0.5,1.5]},
                             "LinearKernel": {"variance": [0.5,1.5]},
                             "RQKernel": {"lengthscale": [0.5,1.5],
                                          "alpha": [0.5, 1.5]},
                             "PeriodicKernel": {"lengthscale": [0.5,1.5],
                                                "period_length": [0.5,1.5]},
                             "ScaleKernel": {"outputscale": [0.5,1.5]},
                             "WhiteNoiseKernel": {'lengthscale': [0.5,1.5]},
                             "MaternKernel": {'lengthscale': [0.5,1.5]},
                             "CosineKernel": {"period_length": [0.5,1.5]}
                             }

    for kernel in get_kernels_in_kernel_expression(kernels):
        hypers = limits[kernel._get_name()]
        for hyperparameter in hypers:
            new_value = torch.rand(1) * (hypers[hyperparameter][1] - hypers[hyperparameter][0]) + hypers[hyperparameter][0]
            setattr(kernel, hyperparameter, new_value)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_text="RBF", weights=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        if kernel_text == "SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_text == "RQ":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        elif kernel_text == "PER":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "PER+SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_text == "PER*SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.RBFKernel())
        elif kernel_text == "PER*LIN":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.LinearKernel())
        elif kernel_text == "MAT32":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)) 
        elif kernel_text == "MAT32*PER":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5) * gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "MAT52":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()) 
        elif kernel_text == "MAT52*PER":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel() * gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "RQ*PER":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel() * gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "RQ*MAT32":
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel() * gpytorch.kernels.MaternKernel(nu=1.5))
        elif kernel_text == "RQ*SE":
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel() * gpytorch.kernels.RBFKernel())


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DataGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_text="SE", weights=None):
        super(DataGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel_text == "SE":
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel_text == "RQ":
            self.covar_module = gpytorch.kernels.RQKernel()
        elif kernel_text == "PER":
            self.covar_module = gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "PER+SE":
            if weights is None:
                self.covar_module = gpytorch.kernels.PeriodicKernel() + gpytorch.kernels.RBFKernel()
            else:
                self.covar_module = weights[0]*gpytorch.kernels.PeriodicKernel() + weights[1]*gpytorch.kernels.RBFKernel()
        elif kernel_text == "PER*SE":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.RBFKernel()
        elif kernel_text == "PER*LIN":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.LinearKernel()
        elif kernel_text == "MAT32":
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5) 
        elif kernel_text == "MAT32*PER":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "MAT52":
            self.covar_module = gpytorch.kernels.MaternKernel() 
        elif kernel_text == "MAT52*PER":
            self.covar_module =  gpytorch.kernels.MaternKernel() * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "RQ*PER":
            self.covar_module = gpytorch.kernels.RQKernel() * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "RQ*MAT32":
            self.covar_module = gpytorch.kernels.RQKernel() * gpytorch.kernels.MaternKernel(nu=1.5) 
        elif kernel_text == "RQ*SE":
            self.covar_module = gpytorch.kernels.RQKernel() * gpytorch.kernels.RBFKernel()
        reparameterize(self.covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




def RMSE(goal, prediction):
    mse = np.mean((goal-prediction)**2)
    rmse = np.sqrt(mse)
    return rmse



def ZScore(data):
    return (data - torch.mean(data)) / torch.std(data)




def draw_store_GP(model, likelihood, train_x, train_y, name):

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = train_x
        outputs = model(test_x)
        predictions = likelihood(outputs)

        mean = predictions.mean
        lower, upper = predictions.confidence_region()

        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.savefig(f"{name}.png")
    return 0





def run_experiment(data_kernel = "PER", model_kernel="SE"):
    """
    This contains the training, kernel search, evaluation, logging, plotting.
    It takes an input file, processes the whole training, evaluation and log
    Returns nothing

    """
    print(f"Data: {data_kernel}; Model: {model_kernel}")
    EXPERIMENT_REPITITIONS = 50

    ### Initialization

    eval_START = -5
    eval_END = 5
    eval_COUNT = 50
    optimizer = "Adam"
    train_iterations = 100
    noise = 0.1

    # training data for model initialization (e.g. 1 point with x=0, y=0) ; this makes initializing the model easier
    prior_x = torch.linspace(0,1,1)
    prior_y = prior_x
    observations_x = torch.linspace(eval_START, eval_END, eval_COUNT)
    X = ZScore(observations_x)

    parameter_results = {}
    # Train loop
    for j in range(EXPERIMENT_REPITITIONS):
        # Generate new data
        # initialize likelihood and model
        data_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        data_model = DataGPModel(prior_x, prior_y, data_likelihood, kernel_text=data_kernel)
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

        Y = ZScore(f_preds.sample())

        # Reinitialize model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X, Y, likelihood, kernel_text=model_kernel)

        likelihood.train()
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(train_iterations):

            optimizer.zero_grad()

            output = model(X)
            loss = -mll(output, Y)
            loss.backward()
            optimizer.step()

        parameter_results[j] = list(model.named_parameters())

        #if j == 0 or j == 25 or j == 49:
        #    draw_store_GP(model, likelihood, X, Y, name=f"{data_kernel}-{model_kernel}-{j}")

    # Write parameter dicts to a file
    with open(f"results/parameter_stuff/data_{data_kernel}-model_{model_kernel}.pickle", "wb+") as f:
         pickle.dump(parameter_results, f)

    ## TODO write filename in FINISHED.log
    #with open("FINISHED.log", "a") as f:
    #    f.writelines(f"data_{data_kernel}-model_{model_kernel}")
    return 0



if __name__ == "__main__":
    with open("FINISHED.log", "r") as f:
        finished_configs = [line.strip() for line in f.readlines()]
    curdir = os.getcwd()
    configs = []
    data_kernels = ["SE", "PER", "MAT32", "MAT52", "RQ", "PER+SE", "PER*SE", "PER*LIN", "MAT32*PER", "MAT52*PER", "RQ*PER", "RQ*SE", "RQ*MAT32"]
    #model_kernels = ["SE", "PER", "SE+PER", "SE*PER", "LIN*PER", "LIN+PER"]
    model_kernels = ["SE", "PER", "PER+SE", "PER*SE", "PER*LIN", "RQ", "RQ*PER", "MAT52", "MAT52*PER", "MAT32", "MAT32*PER"]
    configs = product(data_kernels, model_kernels)
    # Check if the config file is already finished and if it even exists
    for config in configs:
        run_experiment(config[0], config[1])
    #with Pool(processes=7) as pool: # multithreading will lead to problems with the training iterations
    #    pool.map(run_experiment, configs)