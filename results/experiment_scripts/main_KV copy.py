import copy
import dill
import gpytorch


from helpers.gp_classes import DataGPModel, ExactGPModel
from helpers.plotting_functions import plot_training_data, plot_single_input_gp_posterior
from helpers.training_functions import granso_optimization
from helpers.util_functions import prior_distribution, extract_model_parameters

from itertools import product

from laplace_model_selection.metrics import Lap0, LapAIC, LapBIC, AIC, BIC, NestedSampling, MLL, MAP

import matplotlib.pyplot as plt

import os

import torch
import tqdm


def sample_data_from_gp(eval_START, eval_END, eval_COUNT, data_kernel, train_dataset_count=5, test_data=True, test_sample_count=10):
    # training data for model initialization (e.g. 1 point with x=0, y=0) ; this makes initializing the model easier
    prior_x = torch.linspace(0, 1, 1)
    prior_y = prior_x

    # initialize likelihood and model
    data_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    data_model = DataGPModel(prior_x, prior_y, data_likelihood, kernel_name=data_kernel)
    observations_x = torch.linspace(eval_START, eval_END, eval_COUNT)

    # Get into evaluation (predictive posterior) mode
    data_model.eval()
    data_likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.prior_mode(True):
        f_preds = data_model(observations_x)

    all_observations_y = f_preds.sample_n(train_dataset_count)
    if test_data:
        test_observations_y = f_preds.sample_n(test_sample_count)
        return (observations_x, all_observations_y), test_observations_y
    else:
        return (observations_x, all_observations_y)
    



def run_experiment(config):
    """
    This contains the training, kernel search, evaluation, logging, plotting.
    It takes an input file, processes the whole training, evaluation and log
    Returns nothing

    """
    # Load the current settings
    torch.manual_seed(42)
    eval_START = -1 
    eval_END = 1 
    noise_level = torch.sqrt(torch.tensor(config.get("noise_level", 0.0)))
    data_origin = config.get("data_origin", "DataGPModel") #DataGPModel, csv, helpers.data_functions, LODE solution, ...
    eval_COUNT = config["num_data"]
    data_kernel = config["data_kernel"]
    model_kernel = config["model_kernel"]
    

    #logables = dict()
    # Make an "attributes" dictionary containing the settings
    attributes = {
        "dataStart": str(eval_START),
        "dataEnd": str(eval_END),
        "dataNum": str(eval_COUNT),
        "dataGenKernel": data_kernel,
        "modelKernel": model_kernel,
        "dataOrigin": data_origin,
        "noiseLevel": str(noise_level),
    }
    #logables["attributes"] = attributes
    #logables["results"] = list() 
    # Create the data
    if data_origin == "DataGPModel":
        (observations_x, all_observations_y), test_observations_y = sample_data_from_gp(eval_START=eval_START, eval_END=eval_END, eval_COUNT=eval_COUNT, data_kernel=data_kernel, train_dataset_count=10, test_data=True)
    
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
        experiment_path = os.path.join("..", "results", f"kernel_variation_{data_origin}", f"{eval_COUNT}_{data_kernel}", f"{model_kernel}")

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

        # store the first, middle and last test samples
        for test_data_num in [0, 5, 9]:
            # Plot examples of the test data
            f, ax = plot_training_data(original_observations_x, test_observations_y[test_data_num], return_fig=True, show=False)
            f.savefig(os.path.join(experiment_path, f"Test_data_{test_data_num}.png"))
            #f.savefig(os.path.join(experiment_path, f"Test_data_{test_data_num}.pgf"))


        # Create the model and likelihood for MLL
        model_likelihood_MLL = gpytorch.likelihoods.GaussianLikelihood()
        model_MLL = ExactGPModel(observations_x, observations_y, model_likelihood_MLL, kernel_name=model_kernel)

        # Create a version for MAP
        model_likelihood_MAP = gpytorch.likelihoods.GaussianLikelihood()
        model_MAP = ExactGPModel(observations_x, observations_y, model_likelihood_MAP, kernel_name=model_kernel)

        # Train the model with MLL
        neg_scaled_mll, model_MLL, model_likelihood_MLL, training_log_MLL = granso_optimization(model_MLL, model_likelihood_MLL, observations_x, observations_y, random_restarts=5, maxit=1000, MAP=False, double_precision=False, verbose=False)
        dill.dump(training_log_MLL, open(os.path.join(experiment_path, f"{exp_num}_trainLogMLL.pkl"), "wb"))
        dill.dump(copy.deepcopy(model_MLL.state_dict()), open(os.path.join(experiment_path, f"{exp_num}_stateDictMLL.pkl"), "wb"))
        #exp_num_result_dict["training log MLL"] = training_log_MLL

        #exp_num_result_dict["state dict model MLL"] = copy.deepcopy(model_MLL.state_dict())
        pos_unscaled_mll = -neg_scaled_mll*len(observations_x)
        
        # Prior distriibution for Laplace, nested and MAP
        model_parameter_prior = prior_distribution(model_MAP, param_specs=None, kernel_param_specs=None, default_mean=0.0, default_std=10.0)


        # Train the model with MAP
        neg_scaled_map, model_MAP, model_likelihood_MAP, training_log_MAP = granso_optimization(model_MAP, model_likelihood_MAP, observations_x, observations_y, random_restarts=5, maxit=1000, MAP=True, double_precision=False, verbose=False, model_parameter_prior=model_parameter_prior)
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
        nested_sampling = NestedSampling(model=model_MAP, prior=model_parameter_prior, store_full=True, logging=True, pickle_directory=experiment_path, pickle_name=f"nested_sampling_{exp_num}.pkl", maxcall=1e+6, maxiter=1e+6)# print_progress=True, 

        #target_metrics = [map, mll, lap0, lapA, lapB, aic, bic, nested_sampling]
        target_metrics = [mll]

        model_parameters_lap = [p for p in model_MAP.parameters() if p.requires_grad]

        map_call = lambda : map(model_MAP, model_likelihood_MAP, observations_x, observations_y, prior=model_parameter_prior, logging=True)
        mll_call = lambda : mll(model_MLL, model_likelihood_MLL, observations_x, observations_y, logging=True)
        lap0_call = lambda : lap0(pos_unscaled_map, model_parameters_lap, logging=True, model=model_MAP, use_finite_difference_hessian=True)
        lapA_call = lambda: lapA(pos_unscaled_map, model_parameters_lap, logging=True, model=model_MAP, use_finite_difference_hessian=True)
        lapB_call = lambda : lapB(pos_unscaled_map, model_parameters_lap, logging=True, model=model_MAP, use_finite_difference_hessian=True)
        aic_call = lambda : aic(pos_unscaled_mll, len(model_parameters_lap), logging=True)
        bic_call = lambda : bic(pos_unscaled_mll, len(model_parameters_lap), logging=True)
        nested_sampling_call = lambda : nested_sampling(logging=True)

        #metric_calls = [map_call, mll_call, lap0_call, lapA_call, lapB_call, aic_call, bic_call, nested_sampling_call]
        metric_calls = [mll_call]

        for metric, metric_call in zip(target_metrics, metric_calls):
            # instantiate the metric
            #exp_num_result_dict[str(metric)] = dict()
            metric_log = {}
            result, logs = metric_call()
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
        for test_y in test_observations_y:
            model_MLL.set_train_data(observations_x, test_y)
            gpytorch_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(model_MLL.likelihood, model_MLL)
            test_likelihoods.append(gpytorch_log_likelihood(model_MLL(observations_x), test_y))

            model_MAP.set_train_data(observations_x, test_y)
            gpytorch_log_likelihood_map_model = gpytorch.mlls.ExactMarginalLogLikelihood(model_MAP.likelihood, model_MAP)
            test_likelihoods_map.append(gpytorch_log_likelihood_map_model(model_MAP(observations_x), test_y))
        dill.dump(test_likelihoods, open(os.path.join(experiment_path, f"{exp_num}_testLikelihoods.pkl"), "wb")) 
        dill.dump(test_likelihoods_map, open(os.path.join(experiment_path, f"{exp_num}_testLikelihoods_map.pkl"), "wb")) 
        dill.dump(test_observations_y, open(os.path.join(experiment_path, f"{exp_num}_testData.pkl"), "wb")) 
        #logables["results"].append(exp_num_result_dict)
    # Store the logs
    #dill.dump(logables, open(os.path.join(experiment_path, f"results.pkl"), "wb"))


num_data =  [10, 20, 30, 50, 70, 100]
data_kernel = ["LIN", "PER", "MAT32"]

temp = product(num_data, product(data_kernel, data_kernel))
#temp = product(num_data, MI_data_kernel)
configs = [{"num_data": n, "data_kernel": dat[0], "model_kernel": dat[1]} for n, dat in temp]
print(configs)
for config in configs:
    print(config)
    run_experiment(config)