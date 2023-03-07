import copy
import configparser
from experiment_functions import Experiment
from GaussianProcess import ExactGPModel
from globalParams import options
import gpytorch
from helpFunctions import get_string_representation_of_kernel as gsr
from helpFunctions import clean_kernel_expression
from helpFunctions import amount_of_base_kernels
import json
from kernelSearch import *
from matplotlib import pyplot as plt
from metrics import *
from multiprocessing import Pool
import numpy as np
import os
import pdb
import pickle
import random
import tikzplotlib
import time
import torch


def plot_model(model, likelihood, X, Y, return_figure=False, figure=None, 
               ax=None):
    interval_length = torch.max(X) - torch.min(X)
    shift = interval_length * options["plotting"]["border_ratio"]
    test_x = torch.linspace(torch.min(
        X) - shift, torch.max(X) + shift, options["plotting"]["sample_points"])

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    with torch.no_grad():
        if not (figure and ax):
            figure, ax = plt.subplots(1, 1, figsize=(8, 6))

        lower, upper = observed_pred.confidence_region()
        ax.plot(X.numpy(), Y.numpy(), 'k.', zorder=2)
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), color="b", zorder=3)
        amount_of_gradient_steps = 30
        alpha_min = 0.05
        alpha_max = 0.8
        alpha = (alpha_max-alpha_min)/amount_of_gradient_steps
        c = ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(
        ), alpha=alpha+alpha_min, zorder=1).get_facecolor()
        for i in range(1, amount_of_gradient_steps):
            ax.fill_between(test_x.numpy(), (lower+(i/amount_of_gradient_steps)*(upper-lower)).numpy(),
                            (upper-(i/amount_of_gradient_steps)*(upper-lower)).numpy(), alpha=alpha, color=c, zorder=1)
        if options["plotting"]["legend"]:
            ax.plot([], [], 'k.', label="Data")
            ax.plot([], [], 'b', label="Mean")
            ax.plot([], [], color=c, alpha=alpha_max, label="Confidence")
            ax.legend(loc="upper left")
        ax.set_xlabel("Normalized Input")
        ax.set_ylabel("Normalized Output")
    if not return_figure:
        plt.show()
    else:
        return figure, ax


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_text="RBF", weights=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        if kernel_text == "C*C*RBF":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
        elif kernel_text == "C*RBF":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel())
        elif kernel_text == "4C*SIN":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(
            )) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "C*SIN + C*SIN + C*SIN":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) + gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PeriodicKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "C*SIN + C*SIN":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(
            )) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "C*SIN":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "SIN*RBF":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def load_config(config_file):
    with open(os.path.join("configs", config_file), "r") as configfile:
        temp = configfile.readlines()
        text = "".join(i.replace("\n", "") for i in temp)
    var_dict = json.loads(text)

    # Create an experiment name based on the experiment setup
    # e.g. "CKS_Laplace_LR-0.1_SIN-RBF" etc.
    # name_vars = ["Metric", "Kernel_search", "Data_kernel", "LR", "Noise"]
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
    metrics = ["Laplace"]  # ["AIC", "Laplace", "MLL", "MC"]

    eval_START = -5
    eval_END = 5
    eval_COUNT = 100
    optimizer = "Adam"
    train_iterations = 200
    LR = 0.1
    # noise =
    data_scaling = True
    use_BFGS = True
    num_draws = 1000
    parameter_punishment = 2.0

    # set training iterations to the correct config
    options["training"]["max_iter"] = int(train_iterations)

    # Data generation process
    observations_x = torch.linspace(eval_START, eval_END, eval_COUNT)
    if config == "PER":
        observations_y = torch.sin(observations_x * 2*np.pi)
    elif config == "4PER":
        observations_y = torch.sin(observations_x * 2*np.pi) + 0.5*torch.sin(observations_x * 3*np.pi) + \
            0.2*torch.sin(observations_x * 7*np.pi) + 0.1 * \
            torch.sin(observations_x * 11*np.pi)
    elif config == "RBF_PER":
        observations_y = torch.sin(
            observations_x * 2*np.pi) * torch.exp(0.3*observations_x)

    observations_x = (observations_x - torch.mean(observations_x)
                      ) / torch.std(observations_x)
    observations_y = (observations_y - torch.mean(observations_y)
                      ) / torch.std(observations_y)

    logables = dict()
    # Make an "attributes" dictionary containing the settings
    attributes = {
        "eval_START": eval_START,
        "eval_END": eval_END,
        "eval_COUNT": eval_COUNT,
        "optimizer": optimizer,
        "train_iters": train_iterations,
        "LR": LR,
        "BFGS": use_BFGS,
        "data_gen": config
    }
    logables["attributes"] = attributes

    for metric in metrics:
        logables[metric] = dict()

    EXPERIMENT_REPITITIONS = 100
    for exp_num in range(EXPERIMENT_REPITITIONS):
        model_kernels = ["4C*SIN"]
        """
        ["SIN*RBF", "C*C*RBF",
        "C*RBF",
        "4C*SIN",
        "C*SIN + C*SIN + C*SIN",
        "C*SIN + C*SIN",
        "C*SIN"
        ]
        """

        for model_kernel in model_kernels:
            print("\n###############")
            print(model_kernel)
            print("###############\n")
            # Initialize the model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # list_of_variances = [float(variance_list_variance) for i in range(28)]

            model = ExactGPModel(
                observations_x, observations_y, likelihood, model_kernel)

            # Perform a training for AIC and Laplace
            model.train()
            likelihood.train()

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for i in range(train_iterations):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(observations_x)
                # Calc loss and backprop gradients
                loss = -mll(output, observations_y)
                loss.backward()
                optimizer.step()

            if use_BFGS:
                # Additional BFGS optimization to better ensure optimal parameters
                # LBFGS_optimizer = torch.optim.LBFGS(model.parameters(), max_iter=50, line_search_fn='strong_wolfe')
                LBFGS_optimizer = torch.optim.LBFGS(
                    model.parameters(), line_search_fn='strong_wolfe')
                # define closure

                def closure():
                    LBFGS_optimizer.zero_grad()
                    output = model(observations_x)
                    loss = -mll(output, observations_y)
                    LBFGS_optimizer.zero_grad()
                    loss.backward()
                    return loss
                LBFGS_optimizer.step(closure)

            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(observations_x)
            # Calc loss and backprop gradients
            loss = -mll(output, observations_y)

#
            # model.eval()
            # likelihood.eval()
            # with torch.no_grad(), gpytorch.settings.prior_mode(True):
            #    observed_pred_prior = likelihood(model(observations_x))
            #    f_preds = model(observations_x)
            #    mean_posterior = observed_pred_prior.mean
            #    lower_posterior, upper_posterior = observed_pred_prior.confidence_region()

            # mean_y = model(observations_x).mean

            # f, ax = plt.subplots()
            # f, ax = plot_model(model, likelihood, observations_x, observations_y, True, f, ax)
            # ax.plot(observations_x, observations_y, 'k*')
            # image_time = time.time()
            # Store the plots as .png
            # f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}.png"))
            # Store the plots as .tex
            # tikzplotlib.save(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}.tex"))

            if "MLL" in metrics:
                MLL_logs = dict()
                MLL_logs["loss"] = -loss * len(observations_x)
                logables["MLL"][model_kernel] = MLL_logs

            if "AIC" in metrics:
                AIC_approx, AIC_log = calculate_AIC(-loss * len(observations_x), sum(
                    p.numel() for p in model.parameters() if p.requires_grad))
                if torch.isnan(AIC_approx):
                    import pdb
                    pdb.set_trace()
                AIC_logs = dict()
                AIC_logs["loss"] = AIC_approx
                AIC_logs["details"] = AIC_log
                AIC_logs["parameter_punishment"] = parameter_punishment
                logables["AIC"][model_kernel] = AIC_logs

            if "Laplace" in metrics:
                laplace_approx, LApp_log = calculate_laplace(
                    model, loss, param_punish_term=parameter_punishment)
                Laplace_logs = dict()
                Laplace_logs["parameter_punishment"] = parameter_punishment
                Laplace_logs["loss"] = laplace_approx
                Laplace_logs["details"] = LApp_log
                logables["Laplace"][model_kernel] = Laplace_logs

            # Perform MCMC
            if "MC" in metrics:
                MCMC_approx, MC_log = calculate_mc_STAN(
                    model, likelihood, num_draws)
                MC_logs = dict()
                MC_logs["loss"] = MCMC_approx
                MC_logs["num_draws"] = num_draws
                MC_logs["details"] = MC_log
                logables["MC"][model_kernel] = MC_logs

        experiment_path = os.path.join("results", "hardcoded", config)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        with open(os.path.join(experiment_path, f"rest.pickle"), 'wb') as fh:
            pickle.dump(logables, fh)
    return 0


with open("FINISHED.log", "r") as f:
    finished_configs = [line.strip().split("/")[-1] for line in f.readlines()]
curdir = os.getcwd()
keywords = ["4PER"]  # , "PER", "RBF_PER"]
for config in keywords:
    run_experiment(config)