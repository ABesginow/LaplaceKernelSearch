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
# 12. Runtime evaluations
# 13. Calculate Fourier series of a signal and cut after n to generate data
# 14. Vary the values in the variance_list for the different parameters
#     (broader and narrower)

# Not goals of experiments
# - Parameter training

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
            self.covar_module = gpytorch.kernels.CosineKernel()
        elif kernel_text == "SIN+RBF":
            self.covar_module = gpytorch.kernels.CosineKernel() + gpytorch.kernels.RBFKernel()
        elif kernel_text == "SIN*RBF":
            self.covar_module = gpytorch.kernels.CosineKernel() * gpytorch.kernels.RBFKernel()
        elif kernel_text == "SIN*LIN":
            self.covar_module = gpytorch.kernels.CosineKernel() * gpytorch.kernels.LinearKernel()



    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




def monte_carlo_simulate(model, likelihood, number_of_draws=1000, mean=0, std_deviation=2, print_steps=False, scale_data = False):

    # We copy the model to keep the original model unchanged while we assign different parameter values to the copy
    model_mc        = copy.deepcopy(model)
    likelihood_mc   = copy.deepcopy(likelihood)

    # How many parameters does the model have?
    params_list     = [p for p in model_mc.parameters()]
    num_of_params   = len(params_list)

    # We create an array of tensors
    # Each tensor has random values to use for one model parameter
    # These values are normally distributed according to suitable means and standard deviations and not sorted
    random_values   = [None] * num_of_params
    for i in range(num_of_params):
        if (i==0):        # raw noise
            random_values[i] = torch.tensor(  np.random.normal(0.0, 4, number_of_draws)  )
        elif (i==1):      # mean constant
            random_values[i] = torch.tensor(  np.random.normal(0.0, 2, number_of_draws)  )
        elif (i==2):      # raw lengthscale   (or possibly raw offset)
            random_values[i] = torch.tensor(  np.random.normal(0.0, 2, number_of_draws)  )
            #random_values[i] = torch.tensor(  np.random.normal(-2.5, 1.5, number_of_draws)  )
        elif (i==3):      # raw variance      (or possibly raw lengthscale or raw alpha)
            random_values[i] = torch.tensor(  np.random.normal(-2.5, 1.5, number_of_draws)  )
            #random_values[i] = torch.tensor(  np.random.normal(0.0, 2, number_of_draws)  )
        elif (i==4):      # raw period length (or possibly raw lengthscale)
            random_values[i] = torch.tensor(  np.random.normal(2.5, 1.5, number_of_draws)  )
        elif (i==5):      # raw alpha
            random_values[i] = torch.tensor(  np.random.normal(0.0, 2, number_of_draws)  )
        else:
            random_values[i] = torch.tensor(  np.random.normal(mean, std_deviation, number_of_draws)  )

    # An array to store the log-likelihoods in later
    mll_array = []


    for num_draw in range(number_of_draws):
        num_param = 0
        for param_name, param in model_mc.named_parameters():
            param.data.fill_(random_values[num_param][num_draw])
            num_param += 1

        # We can print the parameter values of every round to validate that we are assigning them correctly:
        if print_steps == True:
            print('\nDraw number:  ', num_draw+1)
            print('Randomly assigned parameters:')
            for param_name, param in model_mc.named_parameters():
                print(param.item())

        # Switching to train mode (but we are not actually doing any training)
        model_mc.train()
        likelihood_mc.train()

        mll_mc = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_mc, model_mc)

        output_mc = model_mc(observations_x)

        if scale_data == True:
            loss_mc   = -mll_mc(output_mc, observations_y_transformed)
        else:
            loss_mc   = -mll_mc(output_mc, observations_y)

        # We can look at the loss of our MC model:
        if print_steps == True:
            print('loss of the test copy:       ', loss_mc.item())

        # We save the log-likelihoods of our model into an array
        mll_array.append(num_of_observations * (-loss_mc.item()))

    # We restore the non-logarithmic likelihoods
    # Then we take the mean of these
    # And eventually, the natural logarithm of the mean
    # ... Luckily, Numpy can handle likelihoods of even 10^50 without big issues or inaccuracies.
    #return np.log(np.mean( np.exp(mll_array) ))

    max_mll_array = max(mll_array)
    max_mll_array = np.array(max_mll_array)
    mll_shifted = mll_array-max_mll_array

    return np.log(np.mean(np.exp(mll_shifted))) + max_mll_array




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
    log_name = "..."
    experiment_keyword = var_dict["experiment name"]
    experiment_path = os.path.join("results", f"{experiment_keyword}")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    log_experiment_path = os.path.join(experiment_path, f"{experiment_keyword}.log")
    experiment = Experiment(log_experiment_path, EXPERIMENT_REPITITIONS)

    train_START = -5.
    train_END = 5.
    train_COUNT = 10
    eval_START = -10
    eval_END = 10
    eval_COUNT = 500

    ## Create train data
    # Create base model to generate data

    # training data for model initialization (e.g. 1 point with x=0, y=0) ; this makes initializing the model easier
    prior_x = torch.linspace(0,1,1)
    prior_y = prior_x
    # initialize likelihood and model
    data_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    data_model = ExactGPModel(prior_x, prior_y, data_likelihood, kernel_text="SIN")
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


    X = observations_x[int(0.25*eval_COUNT):int(0.75*eval_COUNT)]
    Y = observations_y[int(0.25*eval_COUNT):int(0.75*eval_COUNT)]

    # Run CKS
    list_of_kernels = [gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())]
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    list_of_variances = [4.0 for i in range(28)]
    model, likelihood = CKS(X, Y, likelihood, list_of_kernels, list_of_variances, experiment, iterations=3, metric="Laplace")




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

    #posterior_plot = experiment.plot_model("Posterior", 1, observations_x, X, Y, lower_posterior, upper_posterior, mean_posterior, ylim=[-3, 3], orig_data=None)
    #posterior_plot.plot()
    f, ax = plt.subplots()
    f, ax = model.plot_model(return_figure=True, figure = f, ax=ax)
    ax.plot(observations_x, observations_y, 'r*')
    plt.show()

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
    KEYWORD = "Laplace"
    configs = os.listdir(os.path.join(curdir, "configs", KEYWORD))
    # Check if the config file is already finished and if it even exists
    configs = [os.path.join(KEYWORD, c) for c in configs if not os.path.join(KEYWORD, c) in finished_configs and os.path.isfile(os.path.join(curdir, "configs", KEYWORD, c))]
    for config in configs:
        run_experiment(config)
    #with Pool(processes=7) as pool:
    #    pool.map(run_experiment, configs)
