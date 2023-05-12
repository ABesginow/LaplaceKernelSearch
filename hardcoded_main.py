import copy
import configparser
from experiment_functions import Experiment
from GaussianProcess import ExactGPModel
from globalParams import options, hyperparameter_limits
import gpytorch
from helpFunctions import get_string_representation_of_kernel as gsr
from helpFunctions import clean_kernel_expression
from helpFunctions import get_kernels_in_kernel_expression
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
        elif kernel_text == "RQ*PER":
            self.covar_module = gpytorch.kernels.RQKernel() * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "RQ*MAT32":
            self.covar_module = gpytorch.kernels.RQKernel() * gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_text == "RQ*SE":
            self.covar_module = gpytorch.kernels.RQKernel() * gpytorch.kernels.RBFKernel()


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
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_text == "RQ":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        elif kernel_text == "PER":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "PER+SE":
            if weights is None:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            else:
                self.covar_module = weights[0]*gpytorch.kernels.PeriodicKernel() + weights[1]*gpytorch.kernels.RBFKernel()
        elif kernel_text == "PER*SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_text == "PER*LIN":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        elif kernel_text == "MAT32":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        elif kernel_text == "MAT32*PER":
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
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
        elif kernel_text == "RQ*PER":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_text == "RQ*MAT32":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        elif kernel_text == "RQ*SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

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

def log_prior(model, theta_mu=None, sigma=None):
    # params -
    # TODO de-spaghettize this once the priors are coded properly
    prior_dict = {'SE': {'raw_lengthscale' : {"mean": -0.21221139138922668 , "std":1.8895426067756804}},
                  'MAT52': {'raw_lengthscale' :{"mean": 0.7993038925994188, "std":2.145122566357853 } },
                  'MAT32': {'raw_lengthscale' :{"mean": 1.5711054238673443, "std":2.4453761235991216 } },
                  'RQ': {'raw_lengthscale' :{"mean": -0.049841950913676276, "std":1.9426354614713097 },
                          'raw_alpha' :{"mean": 1.882148553921053, "std":3.096431944989054 } },
                  'PER':{'raw_lengthscale':{"mean": 0.7778461197268618, "std":2.288946656544974 },
                          'raw_period_length':{"mean": 0.6485334993738499, "std":0.9930632050553377 } },
                  'LIN':{'raw_variance' :{"mean": -0.8017903983055685, "std":0.9966569921354465 } },
                  'c':{'raw_outputscale':{"mean": -1.6253091096349706, "std":2.2570021716661923 } },
                  'noise': {'raw_noise':{"mean": -3.51640656386717, "std":3.5831320474767407 }}}
    #prior_dict = {"SE": {"raw_lengthscale": {"mean": 0.891, "std": 2.195}},
    #              "MAT": {"raw_lengthscale": {"mean": 1.631, "std": 2.554}},
    #              "PER": {"raw_lengthscale": {"mean": 0.338, "std": 2.636},
    #                      "raw_period_length": {"mean": 0.284, "std": 0.902}},
    #              "LIN": {"raw_variance": {"mean": -1.463, "std": 1.633}},
    #              "c": {"raw_outputscale": {"mean": -2.163, "std": 2.448}},
    #              "noise": {"raw_noise": {"mean": -1.792, "std": 3.266}}}

    variances_list = list()
    debug_param_name_list = list()
    theta_mu = list()
    params = list()
    covar_string = gsr(model.covar_module)
    covar_string = covar_string.replace("(", "")
    covar_string = covar_string.replace(")", "")
    covar_string = covar_string.replace(" ", "")
    covar_string = covar_string.replace("PER", "PER+PER")
    covar_string_list = [s.split("*") for s in covar_string.split("+")]
    covar_string_list.insert(0, ["LIKELIHOOD"])
    covar_string_list = list(chain.from_iterable(covar_string_list))
    both_PER_params = False
    for (param_name, param), cov_str in zip(model.named_parameters(), covar_string_list):
        params.append(param.item())
        debug_param_name_list.append(param_name)
        # First param is (always?) noise and is always with the likelihood
        if "likelihood" in param_name:
            theta_mu.append(prior_dict["noise"]["raw_noise"]["mean"])
            variances_list.append(prior_dict["noise"]["raw_noise"]["std"])
            continue
        else:
            if (cov_str == "PER" or cov_str == "RQ") and not both_PER_params:
                theta_mu.append(prior_dict[cov_str][param_name.split(".")[-1]]["mean"])
                variances_list.append(prior_dict[cov_str][param_name.split(".")[-1]]["std"])
                both_PER_params = True
            elif (cov_str == "PER" or cov_str == "RQ") and both_PER_params:
                theta_mu.append(prior_dict[cov_str][param_name.split(".")[-1]]["mean"])
                variances_list.append(prior_dict[cov_str][param_name.split(".")[-1]]["std"])
                both_PER_params = False
            else:
                try:
                    theta_mu.append(prior_dict[cov_str][param_name.split(".")[-1]]["mean"])
                    variances_list.append(prior_dict[cov_str][param_name.split(".")[-1]]["std"])
                except Exception as E:
                    import pdb
                    pdb.set_trace()
                    prev_cov = cov_str
    theta_mu = torch.tensor(theta_mu)
    theta_mu = theta_mu.unsqueeze(0).t()
    sigma = torch.diag(torch.Tensor(variances_list))
    prior = torch.distributions.MultivariateNormal(theta_mu.t(), sigma)

    # for convention reasons I'm diving by the number of datapoints
    return prior.log_prob(torch.Tensor(params)).item() / len(*model.train_inputs)

def optimize_hyperparameters(model, likelihood, train_iterations, X, Y, with_BFGS=False, MAP=False, prior=None):
    """
    find optimal hyperparameters either by BO or by starting from random initial values multiple times, using an optimizer every time
    and then returning the best result
    """
    # setup
    best_loss = 1e400
    optimal_parameters = dict()
    limits = hyperparameter_limits
    # start runs
    for iteration in range(options["training"]["restarts"]+1):
    #for iteration in range(2):
        # optimize and determine loss
        # Perform a training for AIC and Laplace
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(train_iterations):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(X)
            # Calc loss and backprop gradients
            loss = -mll(output, Y)
            if MAP:
                log_p = log_prior(model)
                loss -= log_p
            loss.backward()
            optimizer.step()

        if with_BFGS:
            # Additional BFGS optimization to better ensure optimal parameters
            # LBFGS_optimizer = torch.optim.LBFGS(model.parameters(), max_iter=50, line_search_fn='strong_wolfe')
            LBFGS_optimizer = torch.optim.LBFGS(
                model.parameters(), max_iter=50,
                line_search_fn='strong_wolfe')
            # define closure

            def closure():
                LBFGS_optimizer.zero_grad()
                output = model(X)
                loss = -mll(output, Y)
                if MAP:
                    log_p = log_prior(model)
                    loss -= log_p
                LBFGS_optimizer.zero_grad()
                loss.backward()
                return loss
            LBFGS_optimizer.step(closure)

        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, Y)
        if MAP:
            log_p = log_prior(model)
            loss -= log_p

#        model.train_model(with_BFGS=with_BFGS)
        current_loss = loss
        # check if the current run is better than previous runs
        if current_loss < best_loss:
            # if it is the best, save all used parameters
            best_loss = current_loss
            for param_name, param in model.named_parameters():
                optimal_parameters[param_name] = copy.deepcopy(param)

        # set new random inital values
        model.likelihood.noise_covar.noise = torch.rand(1) * (limits["Noise"][1] - limits["Noise"][0]) + limits["Noise"][0]
        #self.mean_module.constant = torch.rand(1) * (limits["Mean"][1] - limits["Mean"][0]) + limits["Mean"][0]
        for kernel in get_kernels_in_kernel_expression(model.covar_module):
            hypers = limits[kernel._get_name()]
            for hyperparameter in hypers:
                new_value = torch.rand(1) * (hypers[hyperparameter][1] - hypers[hyperparameter][0]) + hypers[hyperparameter][0]
                setattr(kernel, hyperparameter, new_value)

        # print output if enabled
        if options["training"]["print_optimizing_output"]:
            print(f"HYPERPARAMETER OPTIMIZATION: Random Restart {iteration}: loss: {current_loss}, optimal loss: {best_loss}")

    # finally, set the hyperparameters those in the optimal run
    model.initialize(**optimal_parameters)
    output = model(X)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    loss = -mll(output, Y)
    if MAP:
        log_p = log_prior(model)
        loss -= log_p
    if not loss == best_loss:
        import pdb
        pdb.set_trace()
        print(loss)
        print(best_loss)
    return loss


def run_experiment(config):
    """
    This contains the training, kernel search, evaluation, logging, plotting.
    It takes an input file, processes the whole training, evaluation and log
    Returns nothing

    """
    torch.manual_seed(50)
    metrics = ["MLL", "MAP", "AIC", "BIC", "MC", "Laplace", "Laplace_prior"]
    #metrics = ["BIC", "MAP"]
    #metrics = ["Laplace_prior"]
    eval_START = -5
    eval_END = 5
    eval_COUNT = config["num_data"]
    optimizer = "Adam"
    data_kernel = config["data_kernel"]
    train_iterations = 200
    LR = 0.1
    # noise =
    data_scaling = True
    use_BFGS = True
    num_draws = 10000
    parameter_punishment = 2.0

    # set training iterations to the correct config
    options["training"]["max_iter"] = int(train_iterations)


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
        mean_prior = observed_pred_prior.mean
        lower_prior, upper_prior = observed_pred_prior.confidence_region()

    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_covar = f_preds.covariance_matrix
    observations_y = f_preds.sample()           # samples from the model


    experiment_path = os.path.join("results", "hardcoded", f"{eval_COUNT}_{data_kernel}")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    f, ax = plt.subplots()
    ax.plot(observations_x, observations_y, 'k*')
    #Store the plots as .png
    f.savefig(os.path.join(experiment_path, f"DATA.png"))
    #Store the plots as .tex
    tikzplotlib.save(os.path.join(experiment_path, f"DATA.tex"))
    plt.close(f)

    observations_x = (observations_x - torch.mean(observations_x)
                      ) / torch.std(observations_x)
    observations_y = (observations_y - torch.mean(observations_y)
                      ) / torch.std(observations_y)

    f, ax = plt.subplots()
    ax.plot(observations_x, observations_y, 'k*')
    #Store the plots as .png
    f.savefig(os.path.join(experiment_path, f"DATA_normalized.png"))
    #Store the plots as .tex
    tikzplotlib.save(os.path.join(experiment_path, f"DATA_normalized.tex"))
    plt.close(f)
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

    for metric in metrics:
        logables[metric] = dict()

    EXPERIMENT_REPITITIONS = 1
    for exp_num in range(EXPERIMENT_REPITITIONS):

        #model_kernels = ["MAT32+PER"]
        model_kernels = ["C*C*SE", "SE", "PER", "MAT32", "PER*SE", "PER+SE", "MAT32*PER", "MAT32+PER", "MAT32+SE", "MAT32*SE"]
        #model_kernels = ["MAT32*PER"]

        for model_kernel in model_kernels:
            experiment_keyword = f"{model_kernel}_{exp_num}"

            if any([m in metrics for m in ["Laplace", "MLL", "AIC", "BIC"]]):
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
                for i in range(1000):
                    try:
                        train_start = time.time()
                        loss = optimize_hyperparameters(model, likelihood, train_iterations, observations_x, observations_y, use_BFGS)
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
            if "Laplace" in metrics:
                laplace_approx, LApp_log = calculate_laplace(
                    model, (-loss)*len(*model.train_inputs), param_punish_term=parameter_punishment)
                Laplace_logs = dict()
                Laplace_logs["parameter_punishment"] = parameter_punishment
                Laplace_logs["loss"] = laplace_approx
                Laplace_logs["Train time"] = train_time
                Laplace_logs["details"] = LApp_log
                logables["Laplace"][model_kernel] = Laplace_logs

            if any([m in metrics for m in ["Laplace", "MLL", "AIC", "BIC"]]):
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
                ax.plot(observations_x, observations_y, 'k*')
                image_time = time.time()
                #Store the plots as .png
                f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}_MLL.png"))
                #Store the plots as .tex
                tikzplotlib.save(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}_MLL.tex"))
                plt.close(f)

            if "MLL" in metrics:
                MLL_logs = dict()
                MLL_logs["loss"] = -loss * len(observations_x)
                MLL_logs["Train time"] = train_time
                MLL_logs["model parameters"] = list(model.named_parameters())
                logables["MLL"][model_kernel] = MLL_logs

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
                logables["AIC"][model_kernel] = AIC_logs

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
                logables["BIC"][model_kernel] = BIC_logs

            if "MAP" in metrics:
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
                for i in range(1000):
                    try:
                        train_start = time.time()
                        loss = optimize_hyperparameters(model, likelihood, train_iterations, observations_x, observations_y, use_BFGS, MAP=True)
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
                MAP_logs = dict()
                MAP_logs["loss"] = -loss * len(observations_x)
                MAP_logs["Train time"] = train_end - train_start
                MAP_logs["model parameters"] = list(model.named_parameters())
                logables["MAP"][model_kernel] = MAP_logs

                model.eval()
                likelihood.eval()
                with torch.no_grad(), gpytorch.settings.prior_mode(True):
                    observed_pred_prior = likelihood(model(observations_x))
                    f_preds = model(observations_x)
                f, ax = plt.subplots()
                f, ax = plot_model(model, likelihood, observations_x, observations_y, True, f, ax)
                ax.plot(observations_x, observations_y, 'k*')
                image_time = time.time()
                #Store the plots as .png
                f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}_MAP.png"))
                #Store the plots as .tex
                tikzplotlib.save(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}_MAP.tex"))
                plt.close(f)

            # Laplace approximation including prior requires different loss
            if "Laplace_prior" in metrics:
                approx, Lapp_prior_log = calculate_laplace(model, (-loss)*len(*model.train_inputs), with_prior=True)
                Lap_prior_logs = dict()
                Lap_prior_logs["loss"] = approx
                Lap_prior_logs["Train time"] = train_end - train_start
                Lap_prior_logs["details"] = Lapp_prior_log
                logables["Laplace_prior"][model_kernel] = Lap_prior_logs





            # Perform MCMC
            if "MC" in metrics and "MLL" in metrics:
                MCMC_approx, MC_log = calculate_mc_STAN(
                    model, likelihood, num_draws)
                MC_logs = dict()
                MC_logs["loss"] = MCMC_approx
                MC_logs["num_draws"] = num_draws
                MC_logs["details"] = MC_log
                logables["MC"][model_kernel] = MC_logs


    experiment_path = os.path.join("results", "hardcoded",  f"{eval_COUNT}_{data_kernel}")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    with open(os.path.join(experiment_path, f"other_datasizes.pickle"), 'wb') as fh:
        pickle.dump(logables, fh)

    return 0


with open("FINISHED.log", "r") as f:
    finished_configs = [line.strip().split("/")[-1] for line in f.readlines()]
curdir = os.getcwd()
num_data = [70, 30, 20]#[200, 150, 100, 50, 10]
data_kernel = ["SE", "PER", "MAT32", "PER*SE", "PER+SE", "MAT32*PER", "MAT32+PER", "MAT32+SE", "MAT32*SE"]
#data_kernel = ["PER+SE"]
temp = product(num_data, data_kernel)
configs = [{"num_data": n, "data_kernel": dat} for n, dat in temp]
for config in configs:
    run_experiment(config)
