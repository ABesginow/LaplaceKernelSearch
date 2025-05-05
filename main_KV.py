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

from plotting_functions import *
from helper_functions import *
from gp_classes import *

def run_experiment(config, MI=False):
    """
    This contains the training, kernel search, evaluation, logging, plotting.
    It takes an input file, processes the whole training, evaluation and log
    Returns nothing

    """
    torch.manual_seed(42)
    metrics = ["AIC", "BIC", "Laplace", "MLL", "MAP", "Nested"]
    eval_START = -1 
    eval_END = 1 
    eval_COUNT = config["num_data"]
    optimizer = "PyGRANSO"
    data_kernel = config["data_kernel"]
    param_punishments = [0.0, -1.0, "BIC"]

    uninformed = True
    logarithmic_reinit = True

    logables = dict()
    # Make an "attributes" dictionary containing the settings
    attributes = {
        "eval_START": eval_START,
        "eval_END": eval_END,
        "eval_COUNT": eval_COUNT,
        "optimizer": optimizer,
        "uninformed": uninformed,
        "log reinit": logarithmic_reinit,
        "data_gen": data_kernel
    }
    logables["attributes"] = attributes
    logables["results"] = list()
   
    if MI: 
        prior_x = torch.tensor(list(product(torch.linspace(0, 1, 1), torch.linspace(0, 1, 1))))
        prior_y = torch.linspace(0, 1, 1) 
        # initialize likelihood and model
        data_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        data_model = DataMIGPModel(prior_x, prior_y, data_likelihood, kernel_text=data_kernel)
        # Hardcoded variant to just modify the kernel on the second channel
        # This is regexable if I want to in a general fashion
        if "ell" in data_kernel:
            list(data_model.named_parameters())[2][1].data = torch.tensor([[0.5]])

        x_vals = torch.linspace(eval_START, eval_END, eval_COUNT)
        y_vals = torch.linspace(eval_START, eval_END, eval_COUNT)
        xx, yy = torch.meshgrid(x_vals, y_vals)
        observations_x = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        # Get into evaluation (predictive posterior) mode
        data_model.eval()
        data_likelihood.eval()

    else:
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
        f_preds = data_model(observations_x)

    original_observations_x = copy.deepcopy(observations_x)

    EXPERIMENT_REPITITIONS = 5 
    
    all_observations_y = f_preds.sample_n(EXPERIMENT_REPITITIONS)
    test_observations_y = f_preds.sample_n(10)
    for (observations_y, exp_num) in tqdm(zip(all_observations_y, range(EXPERIMENT_REPITITIONS))):
        exp_num_result_dict = dict()
        for metric in metrics:
            exp_num_result_dict[metric] = dict()
        observations_y = torch.round(observations_y, decimals=4)

        # To store performance of kernels on a test dataset (i.e. more samples)
        exp_num_result_dict["test likelihood"] = dict()
        exp_num_result_dict["test likelihood(MAP)"] = dict()
        if not MI:
            observations_x = (observations_x - torch.mean(observations_x)
                            ) / torch.std(observations_x)
        noise_level = torch.sqrt(torch.tensor(0.1))
        original_observations_y = copy.deepcopy(observations_y)
        observations_y = observations_y + torch.randn(observations_y.shape) * noise_level
        #observations_y = (observations_y - torch.mean(observations_y)
        #                  ) / torch.std(observations_y)

        experiment_path = os.path.join("results", "hardcoded", f"{eval_COUNT}_{data_kernel}")

        print(f"PATH: {experiment_path}")

        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        if MI:
            # Head (samples, xx, yy)
            f, ax = plot_3d_gp_samples(original_observations_y, xx, yy, return_figure=True)
        else:
            f, ax = plot_data(original_observations_x, original_observations_y, return_figure=True, title_add="", display_figure=False)
        #Store the plots as .png
        f.savefig(os.path.join(experiment_path, f"DATA_{exp_num}.png"))
        if MI:
            f.savefig(os.path.join(experiment_path, f"DATA_{exp_num}.pgf"))
        else:   
            #Store the plots as .tex
            #tikzplotlib.save(os.path.join(experiment_path, f"DATA_{exp_num}.tex"))
            f.savefig(os.path.join(experiment_path, f"DATA_{exp_num}.pgf"))
        plt.close(f)


        if MI:
            # Head (samples, xx, yy)
            f, ax = plot_3d_gp_samples(observations_y, xx, yy, return_figure=True)
        else:
            f, ax = plot_data(original_observations_x, original_observations_y, return_figure=True, title_add="", display_figure=False)
       #Store the plots as .png
        f.savefig(os.path.join(experiment_path, f"DATA_normalized_{exp_num}.png"))
        if MI:
            f.savefig(os.path.join(experiment_path, f"DATA_normalized_{exp_num}.pgf"))
        else:
            #Store the plots as .tex
            #tikzplotlib.save(os.path.join(experiment_path, f"DATA_normalized_{exp_num}.tex"))
            f.savefig(os.path.join(experiment_path, f"DATA_normalized_{exp_num}.pgf"))
        plt.close(f)

        # store the first, middle and last test samples
        for test_data_num in [0, 5, 9]:
            if MI:
                # Head (samples, xx, yy)
                f, ax = plot_3d_gp_samples(test_observations_y[test_data_num], xx, yy, return_figure=True)
            else:
                f, ax = plot_data(original_observations_x, original_observations_y, return_figure=True, title_add="", display_figure=False)
            f.savefig(os.path.join(experiment_path, f"Test_data_{test_data_num}.png"))

        if MI:
            model_kernels = ["[RBF; RBF]", "[RBF; LIN]", "[LIN; RBF]"]
        else:
            #model_kernels = ["LIN*SE", "LIN*PER", "SE", "SE+SE", "MAT32", "LIN", "PER*SE", "MAT32*PER", "MAT32+PER"]
            model_kernels = ["LIN", "SE", "SE+SE", "MAT32", "LIN*SE", "PER*SE", "MAT32*PER", "MAT32+PER", "LIN*PER", "PER"]

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
                if MI:
                    model = ExactMIGPModel(
                        observations_x, observations_y, likelihood, model_kernel)
                else:
                    model = ExactGPModel(
                        observations_x, observations_y, likelihood, model_kernel)
                for i in range(100):
                    try:
                        train_start = time.time()
                        loss, model, likelihood, mll_train_log = optimize_hyperparameters(
                            model, likelihood,
                            MAP=False,
                            X=observations_x, Y=observations_y, 
                            uninformed=uninformed, logarithmic_reinit=logarithmic_reinit)
                        train_end = time.time()
                        break
                    except Exception as E:
                        import pdb
                        pdb.set_trace()
                        print(f"Error:{E}")
                        print(f"Data:{data_kernel}")
                        print(f"Model:{model_kernel}")
                        model = None
                        if MI:
                            model = ExactMIGPModel(
                                observations_x, observations_y, likelihood, model_kernel)
                        else:
                            model = ExactGPModel(
                                observations_x, observations_y, likelihood, model_kernel)
                        continue
                if loss is np.nan:
                    raise ValueError("training fucked up")
                train_time = train_end - train_start

                print("===========================")
                print(f"Trained parameters (MLL): {list(model.named_parameters())}")
                print("===========================")
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
                try:
                    if MI:
                        #import pdb; pdb.set_trace()
                        print("MLL CASE")
                        f, ax = plot_3d_gp(model, likelihood, x_min=eval_START, x_max=eval_END, y_min=eval_START, y_max=eval_END, resolution=100, return_figure=True)
                        if observations_y.ndim == 1:
                            samples = observations_y.unsqueeze(0)
                        else:
                            samples = observations_y
                        for i, sample in enumerate(samples):
                            z_vals = sample.reshape(xx.shape)
                            ax.plot_surface(xx, yy, z_vals.numpy(), alpha=0.2, color="red")
                    else:
                        model.eval()
                        likelihood.eval()
                        f, ax = plt.subplots()
                        f, ax = plot_model(model, likelihood, observations_x, observations_y, True, f, ax, loss_val = loss, loss_type = "MLL")
                        ax.plot(observations_x, observations_y, 'k*')
                    image_time = time.time()
                    #Store the plots as .png
                    f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_MLL.png"))
                    if MI:
                        f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_MLL.pgf"))
                    else:
                        #Store the plots as .tex
                        #tikzplotlib.save(os.path.join(experiment_path, f"{experiment_keyword}_MLL.tex"))
                        f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_MLL.pgf"))
                    plt.close(f)
                    model.train()
                    likelihood.train()
                except:
                    model.train()
                    likelihood.train()
                    pass

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

            data_model.eval()
            data_likelihood.eval()

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            test_mll = list()
            try:
                for test_sample in test_observations_y:
                    with torch.no_grad(), gpytorch.settings.prior_mode(True):
                        test_sample = test_sample + torch.randn(test_sample.shape) * noise_level
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
                if MI:
                    model = ExactMIGPModel(
                        observations_x, observations_y, likelihood, model_kernel)
                else:
                    model = ExactGPModel(
                        observations_x, observations_y, likelihood, model_kernel)
                for i in range(100):
                    try:
                        train_start = time.time()
                        loss, model, likelihood, map_train_log = optimize_hyperparameters(
                            model, likelihood,
                            MAP=False,
                            X=observations_x, Y=observations_y, 
                            uninformed=uninformed, logarithmic_reinit=logarithmic_reinit)
                        train_end = time.time()
                        break
                    except Exception as E:
                        print(E)
                        model = None
                        if MI:
                            model = ExactMIGPModel(
                                observations_x, observations_y, likelihood, model_kernel)
                        else:
                            model = ExactGPModel(
                                observations_x, observations_y, likelihood, model_kernel)
                        continue
                if loss is np.nan:
                    raise ValueError("training fucked up")
                print("===========================")
                print(f"Trained parameters (MAP): {list(model.named_parameters())}")
                print("===========================")
                if "MAP" in metrics:
                    MAP_logs = dict()
                    MAP_logs["loss"] = -loss * len(observations_x)
                    MAP_logs["Train time"] = train_end - train_start
                    MAP_logs["model parameters"] = list(model.named_parameters())
                    exp_num_result_dict["MAP"][model_kernel] = MAP_logs

                try:
                    if MI:
                        #import pdb;pdb.set_trace()
                        print("MAP CASE")
                        f, ax = plot_3d_gp(model, likelihood, x_min=eval_START, x_max=eval_END, y_min=eval_START, y_max=eval_END, resolution=100, return_figure=True)
                        if observations_y.ndim == 1:
                            samples = observations_y.unsqueeze(0)
                        else:
                            samples = observations_y
                        for i, sample in enumerate(samples):
                            z_vals = sample.reshape(xx.shape)
                            ax.plot_surface(xx, yy, z_vals.numpy(), alpha=0.2, color="red")
                             
                        #f, ax = plot_3d_gp(model, likelihood, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, resolution=100, return_figure=True)
                    else:
                        model.eval()
                        likelihood.eval()
                        f, ax = plt.subplots()
                        f, ax = plot_model(model, likelihood, observations_x, observations_y, True, f, ax, loss_val = loss, loss_type="MAP")
                        ax.plot(observations_x, observations_y, 'k*')
                    image_time = time.time()
                    #Store the plots as .png
                    f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_MAP.png"))
                    if MI:
                        f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_MAP.pgf"))
                    else:
                        #Store the plots as .tex
                        #tikzplotlib.save(os.path.join(experiment_path, f"{experiment_keyword}_MAP.tex"))
                        f.savefig(os.path.join(experiment_path, f"{experiment_keyword}_MAP.pgf"))
                    plt.close(f)
                    model.train()
                    likelihood.train()
                except:
                    model.train()
                    likelihood.train()

            # Laplace approximation including prior requires different loss
            if "Laplace" in metrics:
                Laps_log = {param_punish : {} for param_punish in param_punishments}
                for parameter_punishment in param_punishments:
                    approx, Lap_log = calculate_laplace(model, (-loss)*len(*model.train_inputs), param_punish_term = parameter_punishment, use_finite_difference_hessian=True, uninformed=True)
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
                for test_sample in test_observations_y:
                    with torch.no_grad(), gpytorch.settings.prior_mode(True):
                        test_sample = test_sample + torch.randn(test_sample.shape) * noise_level
                    model.set_train_data(observations_x, test_sample)
                    test_mll.append(mll(model(observations_x), test_sample))
            except Exception as E:
                print(E)
                print("----")
                test_mll = [np.nan]
            exp_num_result_dict["test likelihood(MAP)"][model_kernel] = test_mll


            if "Nested" in metrics:
                model.train()
                likelihood.train()
                #logz_nested, nested_log = NestedSampling(model, store_full=True, pickle_directory=experiment_path, maxcall=3000000)
                logz_nested, nested_log = NestedSampling(model, store_full=True, pickle_directory=experiment_path, maxcall=30000)
                nested_logs = dict()
                nested_logs["loss"] = logz_nested
                nested_logs["details"] = nested_log
                exp_num_result_dict["Nested"][model_kernel] = nested_logs



        logables["results"].append(exp_num_result_dict)

    experiment_path = os.path.join("results", "hardcoded",  f"{eval_COUNT}_{data_kernel}")
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    with open(os.path.join(experiment_path, f"results.pickle"), 'wb') as fh:
        pickle.dump(logables, fh)


num_data =  [10, 20, 30, 50] #[20, 50, 70, 100, 200, 5, 10, 30] 
data_kernel = ["LIN", "SE", "SE+SE", "MAT32", "LIN*SE", "PER*SE", "MAT32*PER", "MAT32+PER", "LIN*PER", "PER"]
#MI_data_kernel = ["[RBF; RBF]", "[RBF; RBF_ell2]", "[RBF; LIN]", "[LIN; RBF]"]

temp = product(num_data, data_kernel)
#temp = product(num_data, MI_data_kernel)
configs = [{"num_data": n, "data_kernel": dat} for n, dat in temp]
print(configs)
for config in configs:
    print(config)
    run_experiment(config, MI=False)
