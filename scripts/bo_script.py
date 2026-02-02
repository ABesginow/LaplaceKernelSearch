import torch
import copy
from typing import List, Dict, Tuple, Callable
from gpytorch.models import ExactGP
from gpytorch.likelihoods import Likelihood
from botorch.models import MultiTaskGP
from botorch.models.transforms.outcome import Standardize
#from botorch.fit import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective, ConstrainedMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.utils import get_infeasible_cost
from traitlets import default

# My imports
from functools import reduce

import gpytorch

from helpers.gp_classes import DataGPModel, ExactGPModel
from helpers.example_kernels import available
from helpers.plotting_functions import plot_training_data, plot_single_input_gp_posterior
from helpers.training_functions import granso_optimization
from helpers.util_functions import prior_distribution, extract_model_parameters

from laplace_model_selection.metrics import Lap0, LapAIC, LapBIC, AIC, BIC, NestedSampling, MLL, MAP

from math import ceil
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
from plotnine import (
    ggplot, aes, geom_col, geom_point, geom_text,
    coord_flip, labs, theme_minimal, theme, position_nudge, position_stack,
    scale_color_manual, scale_fill_manual
)

import tqdm
from typing import List


# --- Parametrized Function API ---
class ParametrizedFunction:
    def __init__(self, name: str, func: Callable[[dict], Tuple[torch.Tensor, torch.Tensor]], param_config: Dict[str, Dict]):
        self.name = name
        self.func = func
        self.param_config = param_config

    def get_bounds(self):
        return {k: v["bounds"] for k, v in self.param_config.items()}

    def default_params(self):
        return {k: v["default"] for k, v in self.param_config.items()}

    def __call__(self, params: dict):
        return self.func(params)

class CompositeFunction:
    def __init__(self, parts: List[ParametrizedFunction]):
        self.parts = parts

    def get_bounds(self):
        bounds = {}
        for i, func in enumerate(self.parts):
            for key, bound in func.get_bounds().items():
                bounds[f"f{i}_{key}"] = bound
        return bounds

    def __call__(self, flat_params: Dict[str, float]):
        xs, ys = [], []
        for i, func in enumerate(self.parts):
            sub_params = {k[len(f"f{i}_"):]: v for k, v in flat_params.items() if k.startswith(f"f{i}_")}
            x, y = func(sub_params)
            xs.append(x)
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)



# --- Metric Evaluation Wrapper ---
def evaluate_metrics(GPs: List[Tuple[ExactGP, Likelihood]], gp_labels: List[str], bo_metric_fkts: Dict[str, Callable], x_data: torch.Tensor, y_data: torch.Tensor) -> Dict[str, float]:
    all_metrics = {}
    for i, (gp_model, likelihood) in enumerate(GPs):
        # Fit the MLL and MAP models
        # Calculate all metrics (AIC, BIC, MLL, MAP, Laplace metrics)
        # Store those metrics in the dictionary in a form such that the underlying kernel can be extracted from the key

        # Fit model with MLL (original)
        gp_model.set_train_data(x_data, y_data, strict=False)
        gp_model.train()
        likelihood.train()
        mll = ExactMarginalLogLikelihood(likelihood, gp_model)
        # Incluse my training code here for MLL
        neg_scaled_mll, model_MLL, likelihood_MLL, training_log_MLL = granso_optimization(gp_model, likelihood, x_data, y_data, random_restarts=20, maxit=1000, MAP=False, double_precision=False, verbose=False)

        # Fit with MAP (deepcopy)
        gp_copy = copy.deepcopy(gp_model)
        lh_copy = copy.deepcopy(likelihood)
        gp_copy.set_train_data(x_data, y_data, strict=False)
        gp_copy.train()
        lh_copy.train()
        # Include my training code here for MAP
        model_parameter_prior = prior_distribution(gp_copy, param_specs=None, kernel_param_specs=None, default_mean=0.0, default_std=10.0)

        neg_scaled_map, model_MAP, likelihood_MAP, training_log_MAP = granso_optimization(gp_copy, lh_copy, x_data, y_data, random_restarts=20, maxit=1000, MAP=True, double_precision=False, verbose=False, model_parameter_prior=model_parameter_prior)

        pos_unscaled_mll = -neg_scaled_mll*len(x_data)
        pos_unscaled_map = -neg_scaled_map*len(x_data)
        # Now the actual experiment starts
        # UNscaled MAP and MLL
        map_metric = MAP(logarithmic=True, scaling=False)
        mll = MLL(logarithmic=True, scaling=False)
        lap0 = Lap0(prior=model_parameter_prior)
        lapA = LapAIC(prior=model_parameter_prior)
        lapB = LapBIC(num_data=len(x_data), prior=model_parameter_prior)
        aic = AIC()
        bic = BIC(len(x_data))
        nested_sampling = NestedSampling(model=model_MAP, prior=model_parameter_prior, store_full=False, logging=True, maxcall=1e+6, maxiter=1e+6)# print_progress=True, 

        model_parameters_lap = [p for p in model_MAP.parameters() if p.requires_grad]

        map_call = lambda : map_metric(model_MAP, likelihood_MAP, x_data, y_data, prior=model_parameter_prior, logging=True)
        mll_call = lambda : mll(model_MLL, likelihood_MLL, x_data, y_data, logging=True)
        lap0_call = lambda : lap0(pos_unscaled_map, model_parameters_lap, logging=True, model=model_MAP, use_finite_difference_hessian=True)
        lapA_call = lambda: lapA(pos_unscaled_map, model_parameters_lap, logging=True, model=model_MAP, use_finite_difference_hessian=True)
        lapB_call = lambda : lapB(pos_unscaled_map, model_parameters_lap, logging=True, model=model_MAP, use_finite_difference_hessian=True)
        aic_call = lambda : aic(pos_unscaled_mll, len(model_parameters_lap), logging=True)
        bic_call = lambda : bic(pos_unscaled_mll, len(model_parameters_lap), logging=True)
        nested_sampling_call = lambda : nested_sampling(logging=True)


        target_metrics = [map_metric, mll, lap0, lapA, lapB, aic, bic]
        metric_calls = [map_call, mll_call, lap0_call, lapA_call, lapB_call, aic_call, bic_call]

        everything = [call_fkt() for call_fkt in metric_calls]
        model_evidences = {f"{str(metric)}_{gp_labels[i]}": e[0] for metric, e in zip(target_metrics, everything)}
        model_evidences[f"AIC_{gp_labels[i]}"] = model_evidences[f"AIC_{gp_labels[i]}"] * (-0.5)
        model_evidences[f"BIC_{gp_labels[i]}"] = model_evidences[f"BIC_{gp_labels[i]}"] * (-0.5)
        all_logs = {str(metric): e[1] for metric, e in zip(target_metrics, everything)}
        all_metrics.update(model_evidences)


    bo_metrics = {metric_name : metric_fkt(all_metrics) for metric_name, metric_fkt in bo_metric_fkts.items()}

    return bo_metrics

# --- Objective and Constraint Wrappers ---
def bo_objective(metric_dict: Dict[str, float]) -> float:
    return metric_dict["log MAP_MAT32"] - metric_dict["Lap032-Lap052"] - metric_dict["log ML52-log ML32"] - metric_dict["log MAP32-Lap032"]



def bo_constraints(metric_dict: Dict[str, float]) -> List[float]:
    return [
        metric_dict["Lap032-Lap052"] - 0.1,
        metric_dict["log ML52-log ML32"] - 0.1,
        metric_dict["log MAP32-Lap032"] - 0.1,
    ]

# --- Main BO Loop ---
def bayesopt_loop(GPs: List[Tuple[ExactGP, Likelihood]],
                 GP_labels: List[str],
                 bo_metric_fkts: Dict[str, Callable],
                 composite_fn: CompositeFunction,
                 bounds_dict: Dict[str, Tuple[float, float]],
                 num_iterations: int = 10,
                 initial_samples: int = 5):

    param_names = list(bounds_dict.keys())
    bounds = torch.tensor([bounds_dict[k] for k in param_names], dtype=torch.float).T

    X = torch.rand(initial_samples, len(param_names))
    # Scale X to the bounds
    for i, (param_name, bound) in enumerate(bounds_dict.items()):
        X[:, i] = X[:, i] * (bound[1] - bound[0]) + bound[0]
    Y = []
    task_ids = []
    metric_cache = []

    for x in X:
        param_values = dict(zip(param_names, x.tolist()))
        x_data, y_data = composite_fn(param_values)
        metrics = evaluate_metrics(GPs, gp_labels=GP_labels, bo_metric_fkts=bo_metric_fkts, x_data=x_data, y_data=y_data)
        metric_cache.append(metrics)
        obj = bo_objective(metrics)
        Y.append([obj])
        task_ids.append([0])

    X = X.double()
    Y = torch.tensor(Y, dtype=torch.double)
    task_ids = torch.tensor(task_ids, dtype=torch.long)

    X_mt = torch.cat([X, task_ids], dim=1)

    model = MultiTaskGP(
        train_X=X_mt,
        train_Y=Y,
        task_feature=X_mt.shape[1] - 1,
        outcome_transform=Standardize(m=1),
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    #fit_gpytorch_mll(model.likelihood)

    for iter in range(num_iterations):
        sampler = SobolQMCNormalSampler(torch.Size([num_iterations]))

        def obj_wrapper(Y):
            return Y[..., 0]

        def constraint_fn_factory():
            constraints_values = bo_constraints(metric_cache[-1])
            
            constraint_fns = []
            for c in constraints_values:
                # Capture each constraint value in a separate closure
                constraint_fns.append(lambda Y, X=None, c_val=c: Y[..., 0] - (c_val + 1e-8))
        
            return constraint_fns

        constraint_fns = constraint_fn_factory()


        feas_cost = get_infeasible_cost(
            X_mt[:, :-1],
            model,
            objective=lambda Y, X=None: Y[..., 0]
        )

        objective = ConstrainedMCObjective(
            objective=lambda Y, X=None: Y[..., 0],
            constraints=constraint_fns,
            infeasible_cost=feas_cost
        )
        acq = qExpectedImprovement(model, best_f=Y.max(), sampler=sampler, objective=objective)

        candidate, _ = optimize_acqf(
            acq,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=32,
        )

        new_x = candidate.detach().squeeze()
        param_values = dict(zip(param_names, new_x.tolist()))
        x_data, y_data = composite_fn(param_values)
        metrics = evaluate_metrics(GPs, gp_labels=GP_labels, bo_metric_fkts=bo_metric_fkts, x_data=x_data, y_data=y_data)
        metric_cache.append(metrics)
        obj = bo_objective(metrics)

        new_x = new_x.unsqueeze(0).double()
        new_y = torch.tensor([[obj]], dtype=torch.double)
        new_task_idx = torch.zeros(1, 1, dtype=torch.long)
        new_x_mt = torch.cat([new_x, new_task_idx], dim=1)

        X_mt = torch.cat([X_mt, new_x_mt], dim=0)
        Y = torch.cat([Y, new_y], dim=0)

        model.set_train_data(X_mt, obj_wrapper(Y), strict=False)
        #fit_gpytorch_mll(model.likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        for _ in range(100):
            try:
                fit_gpytorch_mll(mll)
                break
            except RuntimeError as e:
                continue
        

        print(f"Parameters: {param_values}")
        print(f"Iteration {iter+1}/{num_iterations}, Objective: {obj:.4f}, Constraint: {bo_constraints(metrics)}")



##############################################################################
# Actual script execution starts here
##############################################################################


# --- Function Definitions ---
# Examples for functions

# f(x) = 0
def f0(params: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(torch.tensor(params["x_min"]), torch.tensor(params["x_max"]), int(params["x_count"]), dtype=torch.float)
    y = torch.zeros_like(x)
    return x, y

# f(x) = x^2
def f1(params: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(torch.tensor(params["x_min"]), torch.tensor(params["x_max"]), int(params["x_count"]), dtype=torch.float)
    y = x ** 2
    return x, y

f0_paramed_func = ParametrizedFunction("f(x) = 0", f0, {
    "x_min": {"bounds": (-10.0, 10.0), "default": -10.0},
    "x_max": {"bounds": (-10.0, 10.0), "default": 10.0},
    "x_count": {"bounds": (1, 100), "default": 10},
})

f1_paramed_func = ParametrizedFunction("f(x) = x^2", f1, {
    "x_min": {"bounds": (-10.0, 10.0), "default": -10.0},
    "x_max": {"bounds": (-10.0, 10.0), "default": 10.0},
    "x_count": {"bounds": (1, 100), "default": 10},
})  

composite_fn = CompositeFunction([f0_paramed_func, f1_paramed_func])

gp_model_list = []

temp_data_x = torch.linspace(0.0, 0.0, steps=1, dtype=torch.float)
temp_data_y = temp_data_x

for kernel_name in ["MAT52", "SE"]:
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x=temp_data_x, train_y=temp_data_y, likelihood=likelihood, kernel_name=kernel_name)
    gp_model_list.append((copy.deepcopy(model), copy.deepcopy(likelihood)))


bounds = composite_fn.get_bounds()

num_iterations = 200
initial_samples = 20 

GP_labels = ["MAT32", "MAT52"]
bo_metric_fkts = {
    "log MAP_MAT32" : lambda metrics: metrics["log ML_MAT32"],
    "Lap032-Lap052" : lambda metrics: metrics["Lap0_MAT32"] - metrics["Lap0_MAT52"],
    "log ML52-log ML32" : lambda metrics: metrics["log ML_MAT52"] - metrics["log ML_MAT32"],
    "log MAP32-Lap032" : lambda metrics: metrics["log MAP_MAT32"] - metrics["Lap0_MAT32"],
}

# Run the Bayesian Optimization loop
bayesopt_loop(gp_model_list,
                GP_labels,
                bo_metric_fkts,
                composite_fn,
                bounds,
                num_iterations=num_iterations,
                initial_samples=initial_samples)
# --- End of script execution ---