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
from metrics import NestedSampling
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
        if kernel_text == "SE":
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel_text == "SE+SE":
            self.covar_module = gpytorch.kernels.RBFKernel() + gpytorch.kernels.RBFKernel()
        elif kernel_text == "LIN":
            self.covar_module = gpytorch.kernels.LinearKernel()
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




def run_experiment(config_file, torch_seed):
    """
    This contains the training, kernel search, evaluation, logging, plotting.
    It takes an input file, processes the whole training, evaluation and log
    Returns nothing

    """
    torch.manual_seed(torch_seed)
    EXPERIMENT_REPITITIONS = 50
    total_time_start = time.time()
    options["kernel search"]["print"] = False
    options["training"]["restarts"] = 2
    ### Initialization
    var_dict = load_config(config_file)

    metric = var_dict["Metric"]
    data_kernel = var_dict["Data_kernel"]
    variance_list_variance = var_dict["Variance_list"]
    eval_START = var_dict["eval_START"]
    eval_END = var_dict["eval_END"]
    eval_COUNT = var_dict["eval_COUNT"]
    train_iterations = var_dict["train_iterations"]
    data_scaling = var_dict["Data_scaling"]
    use_BFGS = var_dict["BFGS"]
    num_draws = var_dict["num_draws"] if metric == "MC" else None
    parameter_punishment = var_dict["parameter_punishment"] if metric == "Laplace" else None

    double_precision = False
    if double_precision:
        torch.set_default_dtype(torch.float64)
    # set training iterations to the correct config
    options["training"]["max_iter"] = int(train_iterations)

    ## Create train data
    # Create base model to generate data

    # training data for model initialization (e.g. 1 point with x=0, y=0) ; this makes initializing the model easier
    prior_x = torch.linspace(0, 1, 1)
    prior_y = prior_x
    # initialize likelihood and model
    data_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    data_model = ExactGPModel(prior_x, prior_y, data_likelihood, kernel_text=data_kernel)
    observations_x = torch.linspace(eval_START, eval_END, eval_COUNT)
    observations_x = (observations_x - torch.mean(observations_x)) / torch.std(observations_x)
    # Get into evaluation (predictive posterior) mode
    data_model.eval()
    data_likelihood.eval()

    # Get predictive distribution 
    with torch.no_grad(), gpytorch.settings.prior_mode(True):
        f_preds = data_model(observations_x)

    noise_level = 0.1
    X = observations_x
    # Z-Score scaling
    if data_scaling:
        X = (X - torch.mean(X)) / torch.std(X)
        #Y = (Y - torch.mean(Y)) / torch.std(Y)

    # Sample EXP_REPITITIONS many data sets
    all_observations_y = f_preds.sample_n(EXPERIMENT_REPITITIONS)
    all_observations_y = all_observations_y + torch.randn(all_observations_y.shape) * torch.tensor(noise_level)
    # Sample 10 test data sets
    test_samples = f_preds.sample_n(10)
    test_samples = test_samples + torch.randn(test_samples.shape) * torch.tensor(noise_level)

    #for (exp_num, Y) in enumerate(all_observations_y):
    for (exp_num, Y) in zip(range(0, EXPERIMENT_REPITITIONS, 1), all_observations_y[:]):
        print(config_file)
        print(f"{metric} - {exp_num}/{EXPERIMENT_REPITITIONS} - {time.strftime('%Y-%m-%d %H:%M', time.localtime())}")

        log_name = "..."
        experiment_keyword = var_dict["experiment name"]
        experiment_path = os.path.join("results", metric, f"{experiment_keyword}")
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        log_experiment_path = os.path.join(experiment_path, f"{experiment_keyword}")
        experiment = Experiment(log_experiment_path, exp_num, attributes=var_dict)

        f, ax = plt.subplots()
        ax.plot(X, Y, 'k*')
        ax.plot(X, Y, '-', color="blue")
        #Store the plots as .png
        f.savefig(os.path.join(experiment_path, f"DATA_normalized_{exp_num}.png"))
        #Store the plots as .tex
        tikzplotlib.save(os.path.join(experiment_path, f"DATA_normalized_{exp_num}.tex"))
        plt.close(f)

        # store the first, middle and last test samples
        for test_data_num in [0, 5, 9]:
            f, ax = plt.subplots()
            ax.plot(X, test_samples[test_data_num], "k*")
            ax.plot(X, test_samples[test_data_num], "-", color="blue")
            f.savefig(os.path.join(experiment_path, f"Test_data_{test_data_num}.png"))

        # Run CKS
        list_of_kernels = [gpytorch.kernels.RBFKernel(),
                           gpytorch.kernels.LinearKernel(),
                           gpytorch.kernels.PeriodicKernel(),
                           gpytorch.kernels.MaternKernel(nu=1.5)]
        #list_of_kernels = [gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
        #                   gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()),
        #                   gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel()),
        #                   gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))]
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

        # Do an MCMC evaluation of the resulting kernel and store the likelihood
        # Perform MCMC
        model.train()
        likelihood.train()
        # Necessary since the sampling will directly modify the model
        model_state_dict = copy.deepcopy(model.state_dict())

        Nested_approx, Nested_log = NestedSampling(model, store_full=True, pickle_directory=experiment_path, maxcall=3000000)
        experiment.store_result("Nested approx", Nested_approx)
        experiment.store_result("Nested details", Nested_log)

        # Basically reverting back to the model found using CKS including its parameters
        model.load_state_dict(model_state_dict)

        # Calculate p_k(y_n | X) for the found kernel for new datasets y_n from the same kernel
        # This is directly comparable for ALL metrics, even MAP-LApp
        model.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        #print("post mll init")
        #print(list(model.named_parameters()))
        test_mll = list()
        #manual_test_mll = list()
        try:
            for s in test_samples:
                #model.set_train_data(observations_x, s)
                #print("post new data")
                #print(list(model.named_parameters()))
                test_mll.append(mll(model(X), s))
                #manual_test_mll.append(manual_log_like(model, likelihood))
                #print("post mll calc")
                #print(list(model.named_parameters()))
                #test_mll = [mll(model(observations_x), s) * model.train_targets.numel() for test_sample in test_samples]
        except Exception as E:
            print(E)
            print("Test MLL calculation")
            print("----")
            test_mll = [np.nan]
        experiment.store_result("test mll list", test_mll)
        #experiment.store_result("manual test mll list", manual_test_mll)
        experiment.store_result("avg test mll", torch.mean(torch.Tensor(test_mll)))
        #experiment.store_result("avg manual test mll", torch.mean(torch.Tensor(manual_test_mll)))


        try:
            model.set_train_data(X, test_samples[test_mll.index(max(test_mll))])
            f, ax = plt.subplots()
            f, ax = model.plot_model(return_figure=True, figure = f, ax=ax, posterior=True, test_y = test_samples[test_mll.index(max(test_mll))])
            #ax.plot(X, Y, 'k*')
            ax.set_title(f"{gsr(model.covar_module)} - {max(test_mll)}")
            image_time = time.time()
            # Store the plots as .png
            f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}_best_eval.png"))
            # Store the plots as .tex
            tikzplotlib.save(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}_best_eval.tex"))
            plt.close(f)
            model.set_train_data(X, test_samples[test_mll.index(min(test_mll))])
            f, ax = plt.subplots()
            f, ax = model.plot_model(return_figure=True, figure = f, ax=ax, posterior=True, test_y = test_samples[test_mll.index(min(test_mll))])
            ax.set_title(f"{gsr(model.covar_module)} - {min(test_mll)}")
            image_time = time.time()
            # Store the plots as .png
            f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}_worst_eval.png"))
            # Store the plots as .tex
            tikzplotlib.save(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}_worst_eval.tex"))
            plt.close(f)
        except Exception as E:
            print(E)
            print("Best and worst run plotting")
            print("----")
            open(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}.png"), "w+").close()
            open(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}.tex"), "w+").close()

        # Do this with training a couple iterations on the new data
        #options["training"]["max_iter"] = 50
        #test_mll_retrained = list()
        #try:
        #    for test_sample in test_samples:
        #        model.train_targets = test_sample
        #        model.optimize_hyperparameters(with_BFGS=False, with_Adam=True, MAP=False)
        #        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        #        model.eval()
        #        likelihood.eval()
        #        test_mll_retrained.append(mll(model(observations_x), test_sample) * model.train_targets.numel())
        #except Exception as E:
        #    print(E)
        #    print("----")
        #    test_mll_retrained = [np.nan]
        #experiment.store_result("avg test mll retrained", torch.mean(torch.Tensor(test_mll_retrained)))


        try:
            model.set_train_data(X, Y)
            f, ax = plt.subplots()
            f, ax = model.plot_model(return_figure=True, figure = f, ax=ax)
            ax.plot(X, Y, 'k*')
            ax.set_title(gsr(model.covar_module))
            image_time = time.time()
            # Store the plots as .png
            f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}.png"))
            # Store the plots as .tex
            tikzplotlib.save(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}.tex"))
            plt.close(f)
        except Exception as E:
            print(E)
            print("Posterior plotting?")
            print("----")
            open(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}.png"), "w+").close()
            open(os.path.join(experiment_path, f"{experiment_keyword}_{exp_num}.tex"), "w+").close()




        experiment.store_result("model history", model_history)
        experiment.store_result("random seed", torch_seed)
        experiment.store_result("performance history", performance_history)
        experiment.store_result("loss history", loss_history)
        experiment.store_result("final model", gsr(model.covar_module))
        experiment.store_result("parameters", dict(model.named_parameters())) # oder lieber als reinen string?

        experiment.write_results(os.path.join(experiment_path, f"{exp_num}.pickle"))
        print(f"END {metric} - {exp_num}/{EXPERIMENT_REPITITIONS} - {time.strftime('%Y-%m-%d %H:%M', time.localtime())}")
        # TODO write filename in FINISHED.log
    with open("FINISHED.log", "a") as f:
        f.writelines(config_file + "\n")

    m, s = divmod(time.time() - total_time_start, 60)
    h, m = divmod(m, 60)
    print(f"Total time for {config_file}\n{int(h)}:{int(m)}:{int(s)}")
    return 0






if __name__ == "__main__":
    with open("FINISHED.log", "r") as f:
        finished_configs = [line.strip().split("/")[-1] for line in f.readlines()]
    curdir = os.getcwd()
    keywords = ["AIC", "BIC", "MLL",  "Laplace", "MAP", "Nested"]# "Nested" only when there's a lot of time
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

    #torch_seed = random.randint(1, 1e+7)
    #run_experiment(os.path.join(curdir, "configs/Laplace_prior/-480495070506691892.json"), torch_seed)
    #torch_seed = random.randint(1, 1e+7)
    torch_seed = 42
    for config in configs:
        run_experiment(config, torch_seed)

    #with Pool(processes=4) as pool: # multithreading will lead to problems with the training iterations
    #    pool.starmap(run_experiment, [(config, torch_seed) for config in configs])
