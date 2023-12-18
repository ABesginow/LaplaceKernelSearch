import copy
import configparser
from experiment_functions import Experiment
from GaussianProcess import ExactGPModel
from globalParams import options, hyperparameter_limits
import gpytorch
from helpFunctions import get_string_representation_of_kernel as gsr
from helpFunctions import clean_kernel_expression
from helpFunctions import get_full_kernels_in_kernel_expression
from helpFunctions import amount_of_base_kernels
from itertools import product
import json
from kernelSearch import *
from matplotlib import pyplot as plt
from metrics import *
from multiprocessing import Pool
import numpy as np
import os
import pdb
import pickle
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import random
import tikzplotlib
import time
import torch
from tqdm import tqdm


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
            ax.plot([], [], color=c, alpha=1.0, label="Confidence")
            ax.legend(loc="upper left")
        ax.set_xlabel("Normalized Input")
        ax.set_ylabel("Normalized Output")
    if not return_figure:
        plt.show()
    else:
        return figure, ax


class DataGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_text="SE", weights=None):
        super(DataGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        if kernel_text == "C*C*SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
        elif kernel_text == "SE":
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel_text == "RQ":
            self.covar_module = gpytorch.kernels.RQKernel()
        elif kernel_text == "LIN":
            self.covar_module = gpytorch.kernels.LinearKernel()
        elif kernel_text == "PER":
            self.covar_module = gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "SE+SE":
            self.covar_module = gpytorch.kernels.RBFKernel()+gpytorch.kernels.RBFKernel()
        elif kernel_text == "SE*SE":
            self.covar_module = gpytorch.kernels.RBFKernel()*gpytorch.kernels.RBFKernel()
        elif kernel_text == "PER+SE":
            self.covar_module = gpytorch.kernels.PeriodicKernel() +gpytorch.kernels.RBFKernel()
        elif kernel_text == "PER*SE":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.RBFKernel()
        elif kernel_text == "PER*LIN":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.LinearKernel()
        elif kernel_text == "MAT32":
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_text == "MAT32*PER":
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5) * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "MAT32+PER":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) + gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "MAT32*SE":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) * gpytorch.kernels.RBFKernel()
        elif kernel_text == "MAT32+SE":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) + gpytorch.kernels.RBFKernel()
        elif kernel_text == "MAT52":
            self.covar_module = gpytorch.kernels.MaternKernel()
        elif kernel_text == "MAT52*PER":
            self.covar_module =  gpytorch.kernels.MaternKernel() * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "MAT52+SE":
            self.covar_module =  gpytorch.kernels.MaternKernel() + gpytorch.kernels.RBFKernel()
        elif kernel_text == "RQ*PER":
            self.covar_module = gpytorch.kernels.RQKernel() * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "RQ*MAT32":
            self.covar_module = gpytorch.kernels.RQKernel() * gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_text == "RQ*SE":
            self.covar_module = gpytorch.kernels.RQKernel() * gpytorch.kernels.RBFKernel()
        elif kernel_text == "SE+SE+SE":
            self.covar_module = gpytorch.kernels.RBFKernel() + gpytorch.kernels.RBFKernel() + gpytorch.kernels.RBFKernel()
        elif kernel_text == "(SE+RQ)*PER":
            self.covar_module = (gpytorch.kernels.RBFKernel() + gpytorch.kernels.RQKernel()) * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "((LIN+LIN)+MAT32)":
            self.covar_module = (gpytorch.kernels.LinearKernel() + gpytorch.kernels.LinearKernel()) + gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_text == "MAT32+(MAT52*PER)":
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5) + (gpytorch.kernels.MaternKernel() * gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "(MAT32+(SE+PER))":
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5) + (gpytorch.kernels.RBFKernel() + gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "(((MAT32*LIN)+MAT32)+LIN)":
            self.covar_module = (((gpytorch.kernels.MaternKernel(nu=1.5) * gpytorch.kernels.LinearKernel()) + gpytorch.kernels.MaternKernel(nu=1.5)) + gpytorch.kernels.LinearKernel())
        elif kernel_text == "(MAT32+((SE+PER)*SE))":
            self.covar_module = (gpytorch.kernels.MaternKernel(nu=1.5) + ((gpytorch.kernels.RBFKernel() + gpytorch.kernels.PeriodicKernel()) * gpytorch.kernels.RBFKernel()))
        elif kernel_text == "(MAT32+(SE*PER))":
            self.covar_module = (gpytorch.kernels.MaternKernel(nu=1.5) + (gpytorch.kernels.RBFKernel() * gpytorch.kernels.PeriodicKernel()))
        elif kernel_text == "(MAT52+(MAT32+(PER*SE)))":
            self.covar_module = (gpytorch.kernels.MaternKernel() + (gpytorch.kernels.MaternKernel(nu=1.5) + (gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.RBFKernel())))
        elif kernel_text == "(MAT52*(PER+(SE*PER)))":
            self.covar_module = (gpytorch.kernels.MaternKernel() * (gpytorch.kernels.PeriodicKernel() + (gpytorch.kernels.RBFKernel() * gpytorch.kernels.PeriodicKernel())))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_text="RBF", weights=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        if kernel_text == "C*C*SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
        elif kernel_text == "SE":
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel_text == "LIN":
            self.covar_module = gpytorch.kernels.LinearKernel()
        elif kernel_text == "SE+SE":
            self.covar_module =  gpytorch.kernels.RBFKernel() + gpytorch.kernels.RBFKernel()
        elif kernel_text == "RQ":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        elif kernel_text == "PER":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        #elif kernel_text == "PER+SE":
        #    if weights is None:
        #        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #    else:
        #        self.covar_module = weights[0]*gpytorch.kernels.PeriodicKernel() + weights[1]*gpytorch.kernels.RBFKernel()
        elif kernel_text == "PER*SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #elif kernel_text == "PER*LIN":
        #    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        elif kernel_text == "MAT32":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        elif kernel_text == "MAT32+MAT52":
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        #elif kernel_text == "MAT32*PER":
        #    self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "MAT32+PER":
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "MAT32*SE":
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_text == "MAT32+SE":
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_text == "MAT52":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        elif kernel_text == "MAT52*PER":
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "MAT52+SE":
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_text == "SE*SE":
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_text == "(SE+RQ)*PER":
            self.covar_module =  (gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "SE+SE+SE":
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_text == "MAT32+(MAT52*PER)":
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)) + (gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()))

        #elif kernel_text == "RQ*PER":
        #    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        #elif kernel_text == "RQ*MAT32":
        #    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        #elif kernel_text == "RQ*SE":
        #    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

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


def random_reinit(model):
    for i, (param, limit) in enumerate(zip(model.parameters(), [{"Noise": hyperparameter_limits["Noise"]},*[hyperparameter_limits[kernel] for kernel in get_full_kernels_in_kernel_expression(model.covar_module)]])):
        covar_text = gsr(model.covar_module)
        param_name = list(limit.keys())[0]
        new_param_value = torch.randn_like(param) * (limit[param_name][1] - limit[param_name][0]) + limit[param_name][0]
        param.data = new_param_value



# Define the training loop
def optimize_hyperparameters(model, likelihood, **kwargs):
    """
    find optimal hyperparameters either by BO or by starting from random initial values multiple times, using an optimizer every time
    and then returning the best result
    """
    log_param_path = kwargs.get("log_param_path", False)
    log_likelihood = kwargs.get("log_likelihood", False)
    random_restarts = kwargs.get("random_restarts", options["training"]["restarts"]+1)
    line_search = kwargs.get("line_search", False)
    BFGS_iter = kwargs.get("BFGS_iter", 50)
    train_iterations = kwargs.get("train_iterations", 0)
    train_x = kwargs.get("X", model.train_inputs)
    train_y = kwargs.get("Y", model.train_targets)
    with_BFGS = kwargs.get("with_BFGS", False)
    history_size = kwargs.get("history_size", 100)
    MAP = kwargs.get("MAP", True)
    prior = kwargs.get("prior", False)
    granso = kwargs.get("granso", True)
    double_precision = kwargs.get("double_precision", False)

    # Set up the likelihood and model
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #model = GPModel(train_x, train_y, likelihood)

    # Define the negative log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Set up the PyGRANSO optimizer
    opts = pygransoStruct()
    opts.torch_device = torch.device('cpu')
    nvar = getNvarTorch(model.parameters())
    opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
    opts.opt_tol = float(1e-10)
    opts.limited_mem_size = int(100)
    opts.globalAD = True
    opts.double_precision = double_precision
    opts.quadprog_info_msg = False
    opts.print_level = int(0)
    opts.halt_on_linesearch_bracket = False

    # Define the objective function
    def objective_function(model):
        output = model(train_x)
        loss = -mll(output, train_y)
        if MAP:
            # log_normalized_prior is in metrics.py 
            log_p = log_normalized_prior(model)
            loss -= log_p
        return [loss, None, None]

    best_model_state_dict = model.state_dict()
    best_likelihood_state_dict = likelihood.state_dict()

    random_restarts = int(5)
    best_f = np.inf
    for restart in range(random_restarts):
        print(f"pre training parameters: {list(model.named_parameters())}")
        # Train the model using PyGRANSO
        soln = pygranso(var_spec=model, combined_fn=objective_function, user_opts=opts)
        if soln.final.f < best_f:
            best_f = soln.final.f
            best_model_state_dict = model.state_dict()
            best_likelihood_state_dict = likelihood.state_dict()
        print(f"post training (final): {list(model.named_parameters())} w. loss: {soln.final.f} (smaller=better)")
        random_reinit(model)

    model.load_state_dict(best_model_state_dict)
    likelihood.load_state_dict(best_likelihood_state_dict)

    loss = -mll(model(train_x), train_y)
    if MAP:
        log_p = log_normalized_prior(model)
        loss -= log_p

    #print(f"post training (best): {list(model.named_parameters())} w. loss: {soln.best.f}")
    #print(f"post training (final): {list(model.named_parameters())} w. loss: {soln.final.f}")
    
    #print(torch.autograd.grad(loss, [p for p in model.parameters()], retain_graph=True, create_graph=True, allow_unused=True))
    # Return the trained model
    return loss, model, likelihood




def run_experiment(config):
    """
    This contains the training, kernel search, evaluation, logging, plotting.
    It takes an input file, processes the whole training, evaluation and log
    Returns nothing

    """
    torch.manual_seed(43)
    metrics = ["AIC", "BIC", "MC", "Laplace", "MLL", "MAP"]
    eval_START = -5
    eval_END = 5
    eval_COUNT = config["num_data"]
    optimizer = "PyGRANSO"
    data_kernel = config["data_kernel"]
    train_iterations = 100
    LR = 0.1
    # noise =
    data_scaling = True
    use_BFGS = False
    num_draws = 1000
    param_punishments = [0.0, -1.0, "BIC"]
    double_precision = False
    if double_precision:
        torch.set_default_dtype(torch.float64)

    # set training iterations to the correct config
    options["training"]["max_iter"] = int(train_iterations)
    options["training"]["restarts"] = 2

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
        "data_gen": data_kernel
    }
    logables["attributes"] = attributes
    logables["results"] = list()

    # training data for model initialization (e.g. 1 point with x=0, y=0) ; this makes initializing the model easier
    prior_x = torch.linspace(0, 1, 1)
    prior_y = prior_x
    # initialize likelihood and model
    data_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    data_model = DataGPModel(prior_x, prior_y, data_likelihood, kernel_text=data_kernel)
    observations_x = torch.linspace(eval_START, eval_END, eval_COUNT)
    # Get into evaluation (predictive posterior) mode
    data_model.eval()
    data_likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.prior_mode(True):
        observed_pred_prior = data_likelihood(data_model(observations_x))
        f_preds = data_model(observations_x)

    original_observations_x = copy.deepcopy(observations_x)

    EXPERIMENT_REPITITIONS = 50 
    
    all_observations_y = f_preds.sample_n(EXPERIMENT_REPITITIONS)
    for (observations_y, exp_num) in tqdm(zip(all_observations_y, range(EXPERIMENT_REPITITIONS))):
        exp_num_result_dict = dict()
        for metric in metrics:
            exp_num_result_dict[metric] = dict()
        observations_y = torch.round(observations_y, decimals=4)

        # To store performance of kernels on a test dataset (i.e. more samples)
        exp_num_result_dict["test likelihood"] = dict()
        exp_num_result_dict["test likelihood(MAP)"] = dict()
        observations_x = (observations_x - torch.mean(observations_x)
                        ) / torch.std(observations_x)
        noise_level = torch.sqrt(torch.tensor(0.1))
        original_observations_y = copy.deepcopy(observations_y)
        observations_y = observations_y + torch.randn(observations_y.shape) * noise_level
        #observations_y = (observations_y - torch.mean(observations_y)
        #                  ) / torch.std(observations_y)

        experiment_path = os.path.join("results", "hardcoded", f"{eval_COUNT}_{data_kernel}")
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        f, ax = plt.subplots()
        ax.plot(original_observations_x, original_observations_y, 'k*')
        ax.plot(original_observations_x, original_observations_y, '-', color="blue")
        #Store the plots as .png
        f.savefig(os.path.join(experiment_path, f"DATA_{exp_num}.png"))
        #Store the plots as .tex
        #tikzplotlib.save(os.path.join(experiment_path, f"DATA_{exp_num}.tex"))
        plt.close(f)


        f, ax = plt.subplots()
        ax.plot(observations_x, observations_y, 'k*')
        ax.plot(observations_x, observations_y, '-', color="blue")
        #Store the plots as .png
        f.savefig(os.path.join(experiment_path, f"DATA_normalized_{exp_num}.png"))
        #Store the plots as .tex
        #tikzplotlib.save(os.path.join(experiment_path, f"DATA_normalized_{exp_num}.tex"))
        plt.close(f)

        #model_kernels = ["MAT32+PER"]
        model_kernels = ["SE", "SE+SE", "MAT32"]
        #model_kernels = ["C*C*SE", "SE", "PER", "MAT32", "MAT32+SE", "MAT32*SE"]#"PER*SE", "PER+SE", "MAT32*PER", "MAT32+PER",
        #model_kernels = ["MAT32*PER"]

        for model_kernel in tqdm(model_kernels):
            print(f"Data Kernel: {data_kernel}")
            experiment_keyword = f"{model_kernel}_{exp_num}"

            if any([m in metrics for m in ["MLL", "AIC", "BIC"]]):
                loss = np.nan
                print("\n###############")
                print(model_kernel)
                print("###############\n")
                # Initialize the model
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                # list_of_variances = [float(variance_list_variance) for i in range(28)]
                model = None
                model = ExactGPModel(
                    observations_x, observations_y, likelihood, model_kernel)
                for i in range(100):
                    try:
                        train_start = time.time()
                        loss, model, likelihood = optimize_hyperparameters(
                            model, likelihood,
                            train_iterations=train_iterations, MAP=False,
                            X=observations_x, Y=observations_y,
                            double_precision=double_precision)
                        train_end = time.time()
                        break
                    except Exception as E:
                        print(f"Error:{E}")
                        print(f"Data:{data_kernel}")
                        print(f"Model:{model_kernel}")
                        model = None
                        model = ExactGPModel(
                            observations_x, observations_y, likelihood, model_kernel)
                        continue
                if loss is np.nan:
                    raise ValueError("training fucked up")
                train_time = train_end - train_start
            if "Likelihood Laplace" in metrics:
                Laplace_logs = {param_punish : {} for param_punish in param_punishments}
                for parameter_punishment in param_punishments:
                    laplace_approx, LApp_log = calculate_laplace(
                        model, (-loss)*len(*model.train_inputs), likelihood_laplace=True, param_punish_term=parameter_punishment)
                    Laplace_logs[parameter_punishment]["parameter_punishment"] = parameter_punishment
                    Laplace_logs[parameter_punishment]["loss"] = laplace_approx
                    Laplace_logs[parameter_punishment]["Train time"] = train_time
                    Laplace_logs[parameter_punishment]["details"] = LApp_log
                exp_num_result_dict["Laplace"][model_kernel] = Laplace_logs

            if any([m in metrics for m in ["MLL", "AIC", "BIC"]]):
                model.eval()
                likelihood.eval()
                with torch.no_grad(), gpytorch.settings.prior_mode(True):
                    observed_pred_prior = likelihood(model(observations_x))
                    f_preds = model(observations_x)
                    mean_posterior = observed_pred_prior.mean
                    lower_posterior, upper_posterior = observed_pred_prior.confidence_region()

                mean_y = model(observations_x).mean

                f, ax = plt.subplots()
                f, ax = plot_model(model, likelihood, observations_x, observations_y, True, f, ax)
                ax.plot(original_observations_x, observations_y, 'k*')
                image_time = time.time()
                #Store the plots as .png
                #f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}_MLL.png"))
                #Store the plots as .tex
                #tikzplotlib.save(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}_MLL.tex"))
                plt.close(f)

            if "MLL" in metrics:
                MLL_logs = dict()
                MLL_logs["loss"] = -loss * len(observations_x)
                MLL_logs["Train time"] = train_time
                MLL_logs["model parameters"] = list(model.named_parameters())
                exp_num_result_dict["MLL"][model_kernel] = MLL_logs

            if "AIC" in metrics:
                AIC_approx, AIC_log = calculate_AIC(-loss * len(observations_x), sum(
                    p.numel() for p in model.parameters() if p.requires_grad))
                if torch.isnan(AIC_approx):
                    import pdb
                    pdb.set_trace()
                AIC_logs = dict()
                AIC_logs["loss"] = AIC_approx
                AIC_logs["Train time"] = train_time
                AIC_logs["details"] = AIC_log
                exp_num_result_dict["AIC"][model_kernel] = AIC_logs

            if "BIC" in metrics:
                BIC_approx, BIC_log = calculate_BIC(-loss * len(observations_x), sum(
                    p.numel() for p in model.parameters() if p.requires_grad), torch.tensor(len(observations_x)))
                if torch.isnan(BIC_approx):
                    import pdb
                    pdb.set_trace()
                BIC_logs = dict()
                BIC_logs["loss"] = BIC_approx
                BIC_logs["Train time"] = train_time
                BIC_logs["details"] = BIC_log
                exp_num_result_dict["BIC"][model_kernel] = BIC_logs

            # TODO This has to be done once for MLL and once for MAP!!!!
            data_model.eval()
            data_likelihood.eval()

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            test_mll = list()
            try:
                for s in range(10):
                    with torch.no_grad(), gpytorch.settings.prior_mode(True):
                        f_preds = model(torch.linspace(eval_START, eval_END, eval_COUNT))
                        test_sample = f_preds.sample()
                        observations_x = (torch.linspace(eval_START, eval_END, eval_COUNT) - torch.mean(torch.linspace(eval_START, eval_END, eval_COUNT))
                                        ) / torch.std(torch.linspace(eval_START, eval_END, eval_COUNT))
                        noise_level = 0.4
                        test_sample = test_sample + torch.randn(test_sample.shape) * torch.tensor(noise_level)
                    model.set_train_data(observations_x, test_sample)
                    test_mll.append(mll(model(observations_x), test_sample))
            except Exception as E:
                print(E)
                print("----")
                test_mll = [np.nan]
            exp_num_result_dict["test likelihood"][model_kernel] = test_mll

            if "MAP" in metrics or "Laplace" in metrics:
                loss = np.nan
                print("\n###############")
                print(model_kernel)
                print("###############\n")
                # Initialize the model
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                # list_of_variances = [float(variance_list_variance) for i in range(28)]
                model = None
                model = ExactGPModel(
                    observations_x, observations_y, likelihood, model_kernel)
                for i in range(100):
                    try:
                        train_start = time.time()
                        loss, model, likelihood = optimize_hyperparameters(
                            model, likelihood,
                            train_iterations=train_iterations, MAP=True,
                            X=observations_x, Y=observations_y,
                            double_precision=double_precision)
                        train_end = time.time()
                        break
                    except Exception as E:
                        print(E)
                        model = None
                        model = ExactGPModel(
                            observations_x, observations_y, likelihood, model_kernel)
                        continue
                if loss is np.nan:
                    raise ValueError("training fucked up")
                if "MAP" in metrics:
                    MAP_logs = dict()
                    MAP_logs["loss"] = -loss * len(observations_x)
                    MAP_logs["Train time"] = train_end - train_start
                    MAP_logs["model parameters"] = list(model.named_parameters())
                    exp_num_result_dict["MAP"][model_kernel] = MAP_logs

                model.eval()
                likelihood.eval()
                with torch.no_grad(), gpytorch.settings.prior_mode(True):
                    observed_pred_prior = likelihood(model(observations_x))
                    f_preds = model(observations_x)
                f, ax = plt.subplots()
                f, ax = plot_model(model, likelihood, original_observations_x, observations_y, True, f, ax)
                ax.plot(original_observations_x, observations_y, 'k*')
                image_time = time.time()
                #Store the plots as .png
                #f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}_MAP.png"))
                #Store the plots as .tex
                #tikzplotlib.save(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}_MAP.tex"))
                plt.close(f)

            # Laplace approximation including prior requires different loss
            if "Laplace" in metrics:
                Laps_log = {param_punish : {} for param_punish in param_punishments}
                for parameter_punishment in param_punishments:
                    approx, Lap_log = calculate_laplace(model, (-loss)*len(*model.train_inputs), param_punish_term = parameter_punishment)
                    Laps_log[parameter_punishment]["parameter_punishment"] = parameter_punishment
                    Laps_log[parameter_punishment]["loss"] = approx
                    Laps_log[parameter_punishment]["Train time"] = train_end - train_start
                    Laps_log[parameter_punishment]["details"] = Lap_log
                exp_num_result_dict["Laplace"][model_kernel] = Laps_log



            data_model.eval()
            data_likelihood.eval()

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            test_mll = list()
            try:
                for s in range(10):
                    with torch.no_grad(), gpytorch.settings.prior_mode(True):
                        f_preds = model(torch.linspace(eval_START, eval_END, eval_COUNT))
                        test_sample = f_preds.sample()
                        observations_x = (torch.linspace(eval_START, eval_END, eval_COUNT) - torch.mean(torch.linspace(eval_START, eval_END, eval_COUNT))
                                        ) / torch.std(torch.linspace(eval_START, eval_END, eval_COUNT))
                        noise_level = 0.4
                        test_sample = test_sample + torch.randn(test_sample.shape) * torch.tensor(noise_level)
                    model.set_train_data(observations_x, test_sample)
                    test_mll.append(mll(model(observations_x), test_sample))
            except Exception as E:
                print(E)
                print("----")
                test_mll = [np.nan]
            exp_num_result_dict["test likelihood(MAP)"][model_kernel] = test_mll

            # Perform MCMC
            if "MC" in metrics:
                model.train()
                likelihood.train()
                MCMC_approx, MC_log = calculate_mc_STAN(
                    model, likelihood, 1000, log_param_path=True, 
                    log_full_likelihood=True, log_full_posterior=True)
                MC_logs = dict()
                MC_logs["loss"] = MCMC_approx
                MC_logs["num_draws"] = num_draws
                MC_logs["details"] = MC_log
                exp_num_result_dict["MC"][model_kernel] = MC_logs
        logables["results"].append(exp_num_result_dict)

    experiment_path = os.path.join("results", "hardcoded",  f"{eval_COUNT}_{data_kernel}")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    with open(os.path.join(experiment_path, f"results.pickle"), 'wb') as fh:
        pickle.dump(logables, fh)

    return 0


with open("FINISHED.log", "r") as f:
    finished_configs = [line.strip().split("/")[-1] for line in f.readlines()]
curdir = os.getcwd()
num_data =  [30, 50]#5, 10, 20]#, 30, 50, 70, 100]
data_kernel = ["LIN", "SE", "SE+SE"]
#data_kernel = ["SE", "RQ", "MAT32", "MAT52", "SE*SE",
#               "SE+SE", "MAT32+SE", "MAT52+SE", "MAT32*SE", "PER",
#               "PER*SE", "(SE+RQ)*PER", "SE+SE+SE", "MAT32+(MAT52*PER)"]
#data_kernel = ["SE", "RQ", "MAT32", "MAT52", "SE*SE",
#               "SE+SE", "MAT32+SE", "MAT52+SE", "MAT32*SE", "PER",
#               "PER*SE", "(SE+RQ)*PER", "SE+SE+SE", "MAT32+(MAT52*PER)"]
temp = product(num_data, data_kernel)
configs = [{"num_data": n, "data_kernel": dat} for n, dat in temp]
for config in configs:
    run_experiment(config)
