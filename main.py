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
import tikzplotlib
from experiment_functions import Experiment


def calculate_laplace(model, loss_of_model, variances_list=None):

    # Save a list of model parameters and compute the Hessian of the MLL
    params_list = [p for p in model.parameters()]
    mll         = (num_of_observations * (-loss_of_model))
    env_grads   = torch.autograd.grad(mll, params_list, retain_graph=True, create_graph=True)
    hess_params = []
    for i in range(len(env_grads)):
            hess_params.append(torch.autograd.grad(env_grads[i], params_list, retain_graph=True))

    # theta_mu is a vector of parameter priors
    theta_mu = torch.tensor([1 for p in range(len(params_list))]).reshape(-1,1)

    # sigma is a matrix of variance priors
    sigma = []
    if variances_list is None:
        variances_list = [4 for i in range(len(list(model.parameters())))]
    for i in range(len(params_list)):
        line = (np.zeros(len(params_list))).tolist()
        line[i] = variances_list[i]
        sigma.append(line)
    sigma = torch.tensor(sigma)


    params = torch.tensor(params_list).clone().reshape(-1,1)
    hessian = torch.tensor(hess_params).clone()


    # Here comes what's wrapped in the exp-function:
    thetas_added = params+theta_mu
    thetas_added_transposed = (params+theta_mu).reshape(1,-1)
    middle_term = (sigma.inverse()-hessian).inverse()
    matmuls    = torch.matmul( torch.matmul( torch.matmul( torch.matmul(thetas_added_transposed, sigma.inverse()), middle_term ), hessian ), thetas_added )


    # We can calculate by taking the log of the fraction:
    #fraction = 1 / (sigma.inverse()-hessian).det().sqrt() / sigma.det().sqrt()
    #laplace = mll + torch.log(fraction) + (-1/2) * matmuls

    # Conveniently we can also just express the fraction as a sum:
    laplace = mll - (1/2)*torch.log(sigma.det()) - (1/2)*torch.log( (sigma.inverse()-hessian).det() )  + (-1/2) * matmuls

    return laplace



def RMSE(goal, prediction):
    mse = np.mean((goal-prediction)**2)
    rmse = np.sqrt(mse)
    return mse


def load_config(config_file):
    with open(os.path.join("configs", config_file), "r") as configfile:
        text = configfile.readline()
    var_dict = json.loads(text)

    # Create an experiment name based on the experiment setup
    # e.g. "CKS_Laplace_LR-0.1_SIN-RBF" etc.
    name_vars = ["Metric", "Kernel_search", "Data_kernel", "LR", "Noise"]
    experiment_name = "".join([config[key] for key in name_vars])
    var_dict["experiment name"] = experiment_name

    return var_dict



def log(message, logfile="default.log"):
    with open(os.path.join(path, filename), 'w') as temp_file:
        temp_file.write(message)
    return 0







def run_experiment(config_file):
    """
    This contains the training, kernel search, evaluation, logging, plotting.

    """

    EXPERIMENT_REPITITIONS = 20




    ### Initialization
    var_dict = load_config(config_file)
    log_name = "..."
    experiment_keyword = var_dict["experiment name"]
    experiment_path = os.path.join("results", f"{experiment_keyword}")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    log_experiment_path = os.path.join(experiment_path, f"{experiment_keyword}.log")
    experiment = Experiment(log_experiment_path, EXPERIMENT_REPITITIONS)



    ## Create train data



    ### Model selection & Training (Jan)
    # Store the loss/Laplace/... at each iteration of the model selection!
    # Store the selected kernels over time in a parseable way
    # Store the parameters over time
    #
    calculate_laplace(model, loss)

    ### Calculating various metrics
    # Metrics
    # - RMSE?
    #


    eval_rmse = RMSE(test_y.numpy(), mean.numpy())
    experiment.store_result("eval RMSE", eval_rmse)

    ### Plotting


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
