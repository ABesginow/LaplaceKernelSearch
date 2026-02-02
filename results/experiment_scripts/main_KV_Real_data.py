import copy
import dill
import gpytorch
from helpers.example_kernels import available
from helpers.data_functions import sample_data_from_gp
from ucimlrepo import fetch_ucirepo 
  
from datetime import datetime
from sklearn.datasets import fetch_openml
from pathlib import Path

import pandas as pd
import numpy as np

from helpers.gp_classes import DataGPModel, ExactGPModel
from helpers.plotting_functions import plot_training_data, plot_single_input_gp_posterior
from helpers.training_functions import granso_optimization
from helpers.util_functions import prior_distribution

from itertools import product

from laplace_model_selection.metrics import Lap0, LapAIC, LapBIC, AIC, BIC, NestedSampling, MLL, MAP

import matplotlib.pyplot as plt

import os

import time
import torch
import tqdm

torch.set_default_dtype(torch.float64)


def run_experiment(config, **kwargs):
    """
    This contains the training, kernel search, evaluation, logging, plotting.
    It takes an input file, processes the whole training, evaluation and log
    Returns nothing

    """
    # Load the current settings
    torch.manual_seed(42)
    train_START = 0
    train_END = 1 
    train_COUNT = config.get("num_data", 0)
    eval_START = train_START 
    eval_END = train_END + 1
    eval_COUNT = 100

    noise_level = torch.sqrt(torch.tensor(config.get("noise_level", 0.0)))
    data_origin = config.get("data_origin", "DataGPModel") #DataGPModel, csv, helpers.data_functions, LODE solution, ...
    # No num_data means a premade dataset
    data_kernel = config.get("data_kernel", None)
    model_kernel = config["model_kernel"]
    
    only_nested = kwargs.get("only_nested", False)
    print("Log: Configs loaded")

    #logables = dict()
    # Make an "attributes" dictionary containing the settings
    attributes = {
        "dataStart": str(train_START),
        "dataEnd": str(train_END),
        "dataNum": str(train_COUNT),
        "evalStart": str(eval_START),
        "evalEnd": str(eval_END),
        "evalNum": str(eval_COUNT),
        "dataGenKernel": data_kernel,
        "modelKernel": model_kernel,
        "dataOrigin": data_origin,
        "noiseLevel": str(noise_level),
    }
    #logables["attributes"] = attributes
    #logables["results"] = list() 
    # Create the data
    if data_origin == "DataGPModel":
        (observations_x, all_observations_y), (int_eval_pos, int_eval_obs), (app_eval_pos, app_eval_obs) = sample_data_from_gp(train_START=train_START, train_END=train_END, train_COUNT=train_COUNT, eval_START=eval_START, eval_END=eval_END, eval_COUNT=eval_COUNT, data_kernel=data_kernel, train_dataset_count=10, test_data=True, test_dataset_count=10, interleaved_to_appended_ratio=0.5)
        test_data_exists = True
        print("Log: DataGPModel data generated")
    elif "ucimlrepo" in data_origin:
        # Split the repo name from the dataset name
        dataset_name = data_origin.split("_")[1].split("-")[1]
        dataset_id = int(data_origin.split("_")[1].split("-")[0])

        if dataset_id == 360:
            air_quality = fetch_ucirepo(id=dataset_id)


            # data (as pandas dataframes)
            df = air_quality.data.features

            # Combine Date and Time into a datetime object
            # Date format is usually "DD/MM/YYYY", Time is "HH.MM.SS"
            df["Datetime"] = pd.to_datetime(
                df["Date"] + " " + df["Time"], 
                format="%m/%d/%Y %H:%M:%S"
            )

            # Calculate hours since the first timestamp
            df["HoursSinceStart"] = (df["Datetime"] - df["Datetime"].iloc[0]).dt.total_seconds() / 3600

            # Select the target variable (CO in mg/m^3, which is the CO(GT) column in dataset)
            # In AirQuality dataset, CO(GT) is the ground truth
            y = df[["CO(GT)"]].copy()

            # Drop all -200 and NaN values as those are invalid measurements in AirQuality
            y = y[y != -200].dropna()
            observations_y = torch.tensor(np.array(y)).flatten()

            all_observations_y = [observations_y]

            # X will be only the HoursSinceStart column
            observations_x = torch.tensor(np.array(df.loc[y.index, ["HoursSinceStart"]])).flatten()
            test_data_exists = False
            print("Log: UCI Airquality dataset loaded")
            
        
    
    original_observations_x = copy.deepcopy(observations_x)

    # Run the experiment on each data
    for exp_num, observations_y in tqdm.tqdm(enumerate(all_observations_y)):
        #exp_num_result_dict = dict()

        observations_y = torch.round(observations_y, decimals=4)

        # To store performance of kernels on a test dataset (i.e. more samples)
        #exp_num_result_dict["test likelihood"] = []
        #exp_num_result_dict["test likelihood(MAP)"] = []
        observations_x = (observations_x - torch.mean(observations_x)
                        ) / torch.std(observations_x)

        original_observations_y = copy.deepcopy(observations_y)
        observations_y = observations_y + torch.randn(observations_y.shape) * noise_level

        # Normalization
        if False:
            observations_y = (observations_y - torch.mean(observations_y)) / torch.std(observations_y)
            test_observations_y = (test_observations_y - torch.mean(test_observations_y)) / torch.std(test_observations_y)

        experiment_path = Path.cwd() / Path("results") / Path("results") / Path(f"kernel_variation_{data_origin}", f"{train_COUNT}_{data_kernel}", f"{model_kernel}")

        print(f"PATH: {experiment_path}")

        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        dill.dump(attributes, open(os.path.join(experiment_path, f"{exp_num}_attributes.pkl"), "wb"))
        # Plot without normalization
        f, ax = plot_training_data(original_observations_x, original_observations_y, return_fig=True, show=False)
        f.savefig(os.path.join(experiment_path, f"DATA_{exp_num}.png"))
        #f.savefig(os.path.join(experiment_path, f"DATA_{exp_num}.pgf"))
        plt.close(f)

        # Plot with normalization
        f, ax = plot_training_data(observations_x, observations_y, return_fig=True, show=False)
    #Store the plots as .png
        f.savefig(os.path.join(experiment_path, f"DATA_normalized_{exp_num}.png"))
        #f.savefig(os.path.join(experiment_path, f"DATA_normalized_{exp_num}.pgf"))
        plt.close(f)

        if test_data_exists:
            # store the first, middle and last test samples
            for test_data_num in [0, 5, 9]:
                # Plot examples of the test data
                f, ax = plot_training_data(torch.cat((int_eval_pos, app_eval_pos)), torch.cat((int_eval_obs[exp_num], app_eval_obs[exp_num][test_data_num])), return_fig=True, show=False)
                f.savefig(os.path.join(experiment_path, f"Test_data_{test_data_num}.png"))
                #f.savefig(os.path.join(experiment_path, f"Test_data_{test_data_num}.pgf"))


        # Create the model and likelihood for MLL
        model_likelihood_MLL = gpytorch.likelihoods.GaussianLikelihood()
        model_MLL = ExactGPModel(observations_x, observations_y, model_likelihood_MLL, kernel_name=model_kernel)

        # Create a version for MAP
        model_likelihood_MAP = gpytorch.likelihoods.GaussianLikelihood()
        model_MAP = ExactGPModel(observations_x, observations_y, model_likelihood_MAP, kernel_name=model_kernel)

        if not only_nested:
            start_time = time.time()
            # Train the model with MLL
            neg_scaled_mll, model_MLL, model_likelihood_MLL, training_log_MLL = granso_optimization(model_MLL, model_likelihood_MLL, observations_x, observations_y, random_restarts=10, maxit=1000, MAP=False, double_precision=False, verbose=False)
            end_time = time.time()
            MLL_train_time = end_time - start_time
            dill.dump(MLL_train_time, open(os.path.join(experiment_path, f"{exp_num}_trainTimeMLL.pkl"), "wb"))
            dill.dump(training_log_MLL, open(os.path.join(experiment_path, f"{exp_num}_trainLogMLL.pkl"), "wb"))
            dill.dump(copy.deepcopy(model_MLL.state_dict()), open(os.path.join(experiment_path, f"{exp_num}_stateDictMLL.pkl"), "wb"))
            #exp_num_result_dict["training log MLL"] = training_log_MLL

            #exp_num_result_dict["state dict model MLL"] = copy.deepcopy(model_MLL.state_dict())
            pos_unscaled_mll = -neg_scaled_mll*len(observations_x)
            
        # Prior distriibution for Laplace, nested and MAP
        model_parameter_prior = prior_distribution(model_MAP, param_specs=None, kernel_param_specs=None, default_mean=0.0, default_std=10.0)


        # Train the model with MAP
        if not only_nested:
            start_time = time.time()
            neg_scaled_map, model_MAP, model_likelihood_MAP, training_log_MAP = granso_optimization(model_MAP, model_likelihood_MAP, observations_x, observations_y, random_restarts=10, maxit=1000, MAP=True, double_precision=False, verbose=False, model_parameter_prior=model_parameter_prior)
            end_time = time.time()
            MAP_train_time = end_time - start_time
            dill.dump(MAP_train_time, open(os.path.join(experiment_path, f"{exp_num}_trainTimeMAP.pkl"), "wb"))
            dill.dump(training_log_MAP, open(os.path.join(experiment_path, f"{exp_num}_trainLogMAP.pkl"), "wb"))
            dill.dump(copy.deepcopy(model_MAP.state_dict()), open(os.path.join(experiment_path, f"{exp_num}_stateDictMAP.pkl"), "wb"))
            #exp_num_result_dict["training log MAP"] = training_log_MAP
            #exp_num_result_dict["state dict model MAP"] = copy.deepcopy(model_MAP.state_dict())

            pos_unscaled_map = -neg_scaled_map*len(observations_x)

            # Now the actual experiment starts
            # UNscaled MAP and MLL
            map = MAP(logarithmic=True, scaling=False)
            mll = MLL(logarithmic=True, scaling=False)
            lap0 = Lap0(prior=model_parameter_prior)
            lapA = LapAIC(prior=model_parameter_prior)
            lapB = LapBIC(num_data=len(observations_x), prior=model_parameter_prior)
            aic = AIC()
            bic = BIC(len(observations_x))
        nested_sampling = NestedSampling(model=model_MAP, prior=model_parameter_prior, store_full=True, logging=True, pickle_directory=experiment_path, pickle_name=f"nested_sampling_{exp_num}.pkl", maxcall=1e+5, maxiter=1e+5)# print_progress=True, 

        target_metrics = [map, mll, lap0, lapA, lapB, aic, bic, nested_sampling]
        #target_metrics = [map, mll, lap0, lapA, lapB, aic, bic]
        #target_metrics = [nested_sampling]

        model_parameters_lap = [p for p in model_MAP.parameters() if p.requires_grad]

        if not only_nested:
            map_call = lambda : map(model_MAP, model_likelihood_MAP, observations_x, observations_y, prior=model_parameter_prior, logging=True)
            mll_call = lambda : mll(model_MLL, model_likelihood_MLL, observations_x, observations_y, logging=True)
            lap0_call = lambda : lap0(pos_unscaled_map, model_parameters_lap, logging=True, model=model_MAP, use_finite_difference_hessian=True)
            lapA_call = lambda: lapA(pos_unscaled_map, model_parameters_lap, logging=True, model=model_MAP, use_finite_difference_hessian=True)
            lapB_call = lambda : lapB(pos_unscaled_map, model_parameters_lap, logging=True, model=model_MAP, use_finite_difference_hessian=True)
            aic_call = lambda : aic(pos_unscaled_mll, len(model_parameters_lap), logging=True)
            bic_call = lambda : bic(pos_unscaled_mll, len(model_parameters_lap), logging=True)
        nested_sampling_call = lambda : nested_sampling(logging=True)

        metric_calls = [map_call, mll_call, lap0_call, lapA_call, lapB_call, aic_call, bic_call, nested_sampling_call]
        #metric_calls = [map_call, mll_call, lap0_call, lapA_call, lapB_call, aic_call, bic_call]
        #metric_calls = [nested_sampling_call]

        for metric, metric_call in zip(target_metrics, metric_calls):
            # instantiate the metric
            #exp_num_result_dict[str(metric)] = dict()
            metric_log = {}
            start_time = time.time()
            result, logs = metric_call()
            end_time = time.time()
            total_time = end_time - start_time
            if not "Total time" in logs.keys():
                logs.update({"Metric time": end_time - start_time})
                if str(metric) in ["Lap0", "LapAIC", "LapBIC", "log MAP"]:
                    logs.update({"Total time": total_time + MAP_train_time})
                    logs.update({"Train time": MAP_train_time})
                elif str(metric) in ["AIC", "BIC", "log ML"]:
                    logs.update({"Total time": total_time + MLL_train_time})
                    logs.update({"Train time": MLL_train_time})
            #exp_num_result_dict[str(metric)]["logs"] = logs
            #exp_num_result_dict[str(metric)]["model_evidence_approx"] = result
            metric_log["name"] = str(metric)
            metric_log["logs"] = logs
            metric_log["model_evidence_approx"] = result
            dill.dump(metric_log, open(os.path.join(experiment_path, f"{exp_num}_{str(metric)}.pkl"), "wb"))

        # Do some posterior plotting
        
        with torch.no_grad():
            model_MLL.eval()
            model_MAP.eval()
            model_MLL.likelihood.eval()
            model_MAP.likelihood.eval()
            predictive_distribution_MLL = model_MLL(observations_x)
            fig, ax = plot_single_input_gp_posterior(observations_x, observations_y, observations_x, predictive_distribution_MLL.mean, predictive_distribution_MLL.variance, show=False, return_fig=True)
            fig.savefig(os.path.join(experiment_path, f"Posterior_MLL_{exp_num}.png"))

            predictive_distribution_MAP = model_MAP(observations_x)
            fig, ax = plot_single_input_gp_posterior(observations_x, observations_y, observations_x, predictive_distribution_MAP.mean, predictive_distribution_MAP.variance, show=False, return_fig=True)
            fig.savefig(os.path.join(experiment_path, f"Posterior_MAP_{exp_num}.png"))

        model_MLL.train()
        model_MAP.train()
        model_MLL.likelihood.train()
        model_MAP.likelihood.train()
        # Evaluate the test data
        
        # How to evaluate? 
        # Replace the data inside the model and calculate the mll of the data without training.
        # Since the data is sampled from the same distribution, the likelihood for the trained model should be good.
        test_likelihoods = []
        test_likelihoods_map = []
        (observations_x, all_observations_y), (int_eval_pos, int_eval_obs), (app_eval_pos, app_eval_obs) 
        if test_data_exists and not only_nested:
            model_MLL.eval()
            model_MAP.eval()
            if "int_eval_obs" in locals() and int_eval_obs is not None:
                # Do interleaved test data
                # Metrics to test: MLL, MSE
                for test_y in [int_eval_obs[exp_num]]:
                    ## MLL
                    int_test_likelihoods_MAP = list()
                    int_test_likelihoods_MLL = list()
                    # Do MLL via a predictive distribution at eval_pos and then use dist.log_prob(test_y) to get the likelihood
                    with torch.no_grad():
                        int_MLL_preds = model_MLL(int_eval_pos)
                        int_MAP_preds = model_MAP(int_eval_pos)
                        int_normalized_log_prob_MLL = int_MLL_preds.log_prob(test_y)/test_y.numel()
                        int_normalized_log_prob_MAP = int_MAP_preds.log_prob(test_y)/test_y.numel()
                    int_test_likelihoods_MAP.append(int_normalized_log_prob_MAP)
                    int_test_likelihoods_MLL.append(int_normalized_log_prob_MLL)


                    dill.dump(int_test_likelihoods_MLL, open(os.path.join(experiment_path, f"{exp_num}_intTestLikelihoodsMLL.pkl"), "wb")) 
                    dill.dump(int_test_likelihoods_MAP, open(os.path.join(experiment_path, f"{exp_num}_intTestLikelihoodsMAP.pkl"), "wb")) 
                    dill.dump(int_eval_obs, open(os.path.join(experiment_path, f"{exp_num}_intTestData.pkl"), "wb")) 

                    ## MSE
                    int_test_MSEs_MLL = list()
                    int_test_MSEs_MAP = list()
                    with torch.no_grad():
                        int_MLL_preds = model_MLL(int_eval_pos)
                        int_MAP_preds = model_MAP(int_eval_pos)
                        int_MLL_MSE = torch.mean((test_y - int_MLL_preds.mean)**2)
                        int_MAP_MSE = torch.mean((test_y - int_MAP_preds.mean)**2)
                    int_test_MSEs_MLL.append(int_MLL_MSE)
                    int_test_MSEs_MAP.append(int_MAP_MSE)

                    dill.dump(int_test_MSEs_MLL, open(os.path.join(experiment_path, f"{exp_num}_intTestMSEsMLL.pkl"), "wb")) 
                    dill.dump(int_test_MSEs_MAP, open(os.path.join(experiment_path, f"{exp_num}_intTestMSEsMAP.pkl"), "wb")) 

            if "app_eval_obs" in locals() and app_eval_obs is not None:
                # Do appended test data
                # Metrics to test: MLL, MSE

                ## MLL
                app_test_likelihoods_MAP = list()
                app_test_likelihoods_MLL = list()
                for test_y in app_eval_obs:
                    # Do MLL via a predictive distribution at eval_pos and then use dist.log_prob(test_y) to get the likelihood
                    with torch.no_grad():
                        app_MLL_preds = model_MLL(app_eval_pos)
                        app_MAP_preds = model_MAP(app_eval_pos)
                        app_normalized_log_prob_MLL = app_MLL_preds.log_prob(test_y)/test_y.numel()
                        app_normalized_log_prob_MAP = app_MAP_preds.log_prob(test_y)/test_y.numel()
                    app_test_likelihoods_MAP.append(app_normalized_log_prob_MAP)
                    app_test_likelihoods_MLL.append(app_normalized_log_prob_MLL)


                dill.dump(app_test_likelihoods_MLL, open(os.path.join(experiment_path, f"{exp_num}_appTestLikelihoodsMLL.pkl"), "wb")) 
                dill.dump(app_test_likelihoods_MAP, open(os.path.join(experiment_path, f"{exp_num}_appTestLikelihoodsMAP.pkl"), "wb")) 
                dill.dump(app_eval_obs, open(os.path.join(experiment_path, f"{exp_num}_appTestData.pkl"), "wb")) 

                ## MSE
                app_test_MSEs_MLL = list()
                app_test_MSEs_MAP = list()
                for test_y in app_eval_obs:
                    with torch.no_grad():
                        app_MLL_preds = model_MLL(app_eval_pos)
                        app_MAP_preds = model_MAP(app_eval_pos)
                        app_MLL_MSE = torch.mean((test_y - app_MLL_preds.mean)**2, axis=1)
                        app_MAP_MSE = torch.mean((test_y - app_MAP_preds.mean)**2, axis=1)
                    app_test_MSEs_MLL.append(app_MLL_MSE)
                    app_test_MSEs_MAP.append(app_MAP_MSE)

                dill.dump(app_test_MSEs_MLL, open(os.path.join(experiment_path, f"{exp_num}_appTestMSEsMLL.pkl"), "wb")) 
                dill.dump(app_test_MSEs_MAP, open(os.path.join(experiment_path, f"{exp_num}_appTestMSEsMAP.pkl"), "wb")) 



        #logables["results"].append(exp_num_result_dict)
    # Store the logs
    #dill.dump(logables, open(os.path.join(experiment_path, f"results.pkl"), "wb"))


num_data = [200]#[10, 15, 25, 40, 50, 75, 100, 200]# [2, 3, 5, 7, 15, 10, 25, 40, 50, 75, 100]

data_kernel = ["SE", "MAT52", "LIN", "MAT32", "MAT12", "PER"]

#broken_list = ["PER", "C*PER", "MAT32+PER", "PER*(SE+RQ)"]
#finished_list = ["LIN"]
#finished_list = []
#data_kernel = [k for k in available() if not ";" in k and not k in broken_list and not k in finished_list]

temp = product(num_data, product(data_kernel, data_kernel))
configs = [{"data_origin": "DataGPModel", "num_data": n, "data_kernel": dat[0], "model_kernel": dat[1]} for n, dat in temp]

# Running UCI data
#temp = data_kernel
#configs = [{"data_origin":"ucimlrepo_360-AirQuality", "model_kernel": dat} for dat in temp]
#configs = [{"data_origin":"ucimlrepo_NUM-Name", "model_kernel": dat} for dat in temp]

print(configs)
for config in configs:
    print(config)
    try:
        run_experiment(config, only_nested=False)
    except Exception as e:
        print(f"Error in config {config}: {e}")
        continue