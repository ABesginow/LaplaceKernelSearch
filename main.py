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

# Things to do in code
# - Deal with multiple times the same parameter in the MCMC case
#   (Probably look into the parameter names and store a dict of names+values)
# - Deal with the issue of non PD matrices when parametrizations are bad
# 13. Calculate Fourier series of a signal/function and cut after n for data ?
#     Purpose: Gives a linear combination of sine/cosine and "represent" real
#     data
# 12. Runtime evaluations (Logging)
# 3. Reproducibility experiments (Logging/Repetitions)

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
from multiprocessing import Pool
import os
import pdb
import random
import tikzplotlib
import time
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_text="RBF", weights=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        if kernel_text == "RBF":
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel_text == "RQ":
            self.covar_module = gpytorch.kernels.RQKernel()
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
        elif kernel_text == "MAT32":
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5) 
        elif kernel_text == "MAT32*SIN":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_text == "MAT52":
            self.covar_module = gpytorch.kernels.MaternKernel() 
        elif kernel_text == "MAT52*SIN":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.MaternKernel()
        elif kernel_text == "RQ*SIN":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.RQKernel()
        elif kernel_text == "RQ*MAT32":
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5) * gpytorch.kernels.RQKernel()
        elif kernel_text == "RQ*RBF":
            self.covar_module = gpytorch.kernels.RBFKernel() * gpytorch.kernels.RQKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



def RMSE(goal, prediction):
    mse = np.mean((goal-prediction)**2)
    rmse = np.sqrt(mse)
    return rmse


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




def run_experiment(config_file):
    """
    This contains the training, kernel search, evaluation, logging, plotting.
    It takes an input file, processes the whole training, evaluation and log
    Returns nothing

    """

    EXPERIMENT_REPITITIONS = 50 
    for exp_num in range(EXPERIMENT_REPITITIONS):

        ### Initialization
        var_dict = load_config(config_file)

        metric = var_dict["Metric"]
        kernel_search = var_dict["Kernel_search"]
        train_data_ratio = var_dict["train_data_ratio"]
        data_kernel = var_dict["Data_kernel"]
        weights = var_dict["weights"]
        variance_list_variance = var_dict["Variance_list"]
        eval_START = var_dict["eval_START"]
        eval_END = var_dict["eval_END"]
        eval_COUNT = var_dict["eval_COUNT"]
        optimizer = var_dict["optimizer"]
        train_iterations = var_dict["train_iterations"]
        LR = var_dict["LR"]
        noise = var_dict["Noise"]
        data_scaling = var_dict["Data_scaling"]
        use_BFGS = var_dict["BFGS"]
        num_draws = var_dict["num_draws"] if metric == "MC" else None
        parameter_punishment = var_dict["parameter_punishment"] if metric == "Laplace" else None

        # set training iterations to the correct config
        options["training"]["max_iter"] = int(train_iterations)

        print(metric)

        log_name = "..."
        experiment_keyword = var_dict["experiment name"]
        experiment_path = os.path.join("results", metric, f"{experiment_keyword}")
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        log_experiment_path = os.path.join(experiment_path, f"{experiment_keyword}")
        experiment = Experiment(log_experiment_path, exp_num, attributes=var_dict)



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

        X = observations_x  #[int((1-train_data_ratio)*0.5*eval_COUNT):int((1+train_data_ratio)*0.5*eval_COUNT)]
        Y = observations_y  #[int((1-train_data_ratio)*0.5*eval_COUNT):int((1+train_data_ratio)*0.5*eval_COUNT)]

        # Percentage noise
        if noise:
            Y = Y + torch.randn(Y.shape) * (noise * Y.max())
        # Z-Score scaling
        if data_scaling:
            X = (X - torch.mean(X)) / torch.std(X)
            Y = (Y - torch.mean(Y)) / torch.std(Y)



        # Run CKS
        list_of_kernels = [gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
                           gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()),
                           gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel()), 
                           gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()),
                           gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)),
                           gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        list_of_variances = [float(variance_list_variance) for i in range(28)] # ist das richtig so?? Kommt mir falsch vor...
        #try:

        KS_start = time.time()
        model, likelihood, model_history, performance_history, loss_history, logables, explosion_counter, total_counter = CKS(X, Y, likelihood, list_of_kernels, list_of_variances, experiment, iterations=3, metric=metric, BFGS=use_BFGS, num_draws=num_draws, param_punish_term = parameter_punishment)
        KS_end = time.time()
        experiment.store_result("Kernel search time", KS_end - KS_start)
        experiment.store_result("details", logables)
        experiment.store_result("total count", total_counter)
        experiment.store_result("explosion count", explosion_counter)
        # With the exception check in CKS hyperparameter training this shouldn't
        # happen
        #except Exception as e:
        #    experiment.store_result("catastrophic failure", True)
        #    experiment.store_result("Exception", e)
        #    experiment.write_results()
        #    print("CATASTROPHIC FAILURE")
        #    return -1


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
        ax.set_title(gsr(model.covar_module)) 
        image_time = time.time()
        # Store the plots as .png
        f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}.png"))
        # Store the plots as .tex
        #tikzplotlib.save(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}.tex"))



        #eval_rmse = RMSE(observations_y.numpy(), mean_y.detach().numpy())
        #print(eval_rmse)

        #experiment.store_result("eval RMSE", eval_rmse)
        experiment.store_result("model history", model_history)
        experiment.store_result("performance history", performance_history)
        experiment.store_result("loss history", loss_history)
        experiment.store_result("final model", gsr(model.covar_module))
        experiment.store_result("parameters", dict(model.named_parameters())) # oder lieber als reinen string?

        experiment.write_results(os.path.join(experiment_path, f"{exp_num}.pickle"))
        # TODO write filename in FINISHED.log
    with open("FINISHED.log", "a") as f:
        f.writelines(config_file + "\n")
    return 0





if __name__ == "__main__":
    with open("FINISHED.log", "r") as f:
        finished_configs = [line.strip().split("/")[-1] for line in f.readlines()]
    curdir = os.getcwd()
    keywords = ["MLL", "AIC", "Laplace", "Laplace_prior"] # "MC" only when there's a lot of time
    configs = []
    for KEYWORD in keywords:
        configs.extend([os.path.join(curdir, "configs", KEYWORD, item) for item in os.listdir(os.path.join(curdir, "configs", KEYWORD))])
    if any([".DS_Store" in config for config in configs]):
        DS_STORES = [config for config in configs if ".DS_Store" in config]
        for conf in DS_STORES:
            configs.remove(conf)
    # Check if the config file is already finished and if it even exists
    configs = [c for c in configs if not c.split("/")[-1] in finished_configs and os.path.isfile(c)]
    #configs = [c for c in configs if not c.split("/")[-1] in finished_configs and os.path.isfile(c)]

    for config in configs:
        run_experiment(config)

    with open("running.log", "r") as fin, open("running.log", "w+") as fout:
        for line in fin:
            line = line.replace(config.split("/")[-1], "")
            fout.write(line)

    #for config in configs:
    #    run_experiment(config)
    #with Pool(processes=7) as pool: # multithreading will lead to problems with the training iterations
    #    pool.map(run_experiment, configs)
