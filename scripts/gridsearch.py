# %%
from helpers.plotting_functions import plot_3d_gp, plot_3d_data, plot_single_input_gp_posterior
from helpers.util_functions import prior_distribution, reparameterize_model_full, reparameterize_model_trainable, log_normalized_prior, extract_trainable_model_parameters
from laplace_model_selection.metrics import Lap0, LapAIC, LapBIC, AIC, BIC, NestedSampling, MLL, MAP
from helpers.training_functions import granso_optimization
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import torch
import gpytorch
torch.set_default_dtype(torch.float64)

import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import dill
import copy

# %%
X = torch.linspace(0, 10, 100)
y = torch.linspace(0, 10, 100)


base_kernel_parameter_priors = {
    ("RBFKernel", "lengthscale"): {"mean": 0.0, "std": 10.0}, 
    ("MaternKernel", "lengthscale"): {"mean": 0.0, "std": 10.0},
    ("LinearKernel", "variance"): {"mean": 0.0, "std": 10.0},
    ("AffineKernel", "variance"): {"mean": 0.0, "std": 10.0},
    ("RQKernel", "lengthscale"): {"mean": 0.0, "std": 10.0},
    ("RQKernel", "alpha"): {"mean": 0.0, "std": 10.0},
    ("CosineKernel", "period_length"): {"mean": 0.0, "std": 10.0},
    ("PeriodicKernel", "lengthscale"): {"mean": 0.0, "std": 10.0},
    ("PeriodicKernel", "period_length"): {"mean": 0.0, "std": 10.0},
    ("ScaleKernel", "outputscale"): {"mean": 0.0, "std": 10.0},
    ("NSumCSEKernel", "n_sum_scale"): {"mean": 0.0, "std": 10.0},
    ("LODE_Kernel", "signal_variance_2_0"): {"mean": 0.0, "std": 10.0},  # full match
    ("LODE_Kernel", "lengthscale"): {"mean": 0.0, "std": 10.0},           # base fallback
}


base_parameter_priors = {
    "likelihood.raw_task_noises": {"mean": 0.0, "std": 10.0},
    "likelihood.raw_noise": {"mean": 0.0, "std": 10.0}
}


base_kernel_param_specs = {
    ("RBFKernel", "lengthscale"): {"bounds": (-1.0, 10.0), "type":"uniform"}, # add ', "type": "uniform"},' # to use uniform distribution
    ("MaternKernel", "lengthscale"): {"bounds": (-1.0, 10.0), "type":"uniform"},
    ("LinearKernel", "variance"): {"bounds": (1e-1, 1.0)},
    ("AffineKernel", "variance"): {"bounds": (1e-1, 1.0)},
    ("RQKernel", "lengthscale"): {"bounds": (-1.0, 10.0), "type":"uniform"},
    ("RQKernel", "alpha"): {"bounds": (1e-1, 10.0), "type":"uniform"},
    ("CosineKernel", "period_length"): {"bounds": (-1.0, 10.0), "type": "uniform"},
    #("PeriodicKernel", "lengthscale"): {"bounds": (1e-1, 5.0)},
    #("PeriodicKernel", "period_length"): {"bounds": (-1.0, 10.0), "type": "uniform"},
    ("PeriodicKernel", "period_length"): {"bounds": (-1.0, 10.0), "type": "uniform"},
    ("ScaleKernel", "outputscale"): {"bounds": (-1.0, 10.0), "type": "uniform"},
    ("NSumCSEKernel", "n_sum_scale"): {"bounds": (-1.0, 10.0), "type": "uniform"},
    #("LODE_Kernel", "signal_variance_2_0"): {"bounds": (0.05, 0.5)},  # full match
    ("LODE_Kernel", "signal_variance"): {"bounds": (1e-1, 10)},  # base
    ("LODE_Kernel", "lengthscale"): {"bounds": (1e-1, 5.0)},           
}


base_param_specs = {
    "likelihood.raw_task_noises": {"bounds": (1e-1, 1e-0)},
    "likelihood.raw_noise": {"bounds": (1e-0, 1e+1), "type":"uniform"}
}

# %%
from helpers.example_kernels import _c_se, _se
import torch
import gpytorch

class NSumCSEKernel(gpytorch.kernels.Kernel):
    """
    k(x, x') = (sum_i sigma_i) * SE(x, x')
    with fixed SE lengthscale = 1.
    """
    def __init__(self, num_c: int = 1, active_dims=None):
        super().__init__(active_dims=active_dims)

        # Base SE (RBF) kernel
        self.base_kernel = gpytorch.kernels.RBFKernel(active_dims=active_dims)

        # Fix lengthscale to 1.0 and disable gradient
        self.base_kernel.lengthscale = 1.0
        self.base_kernel.raw_lengthscale.requires_grad_(False)

        self.num_sigmas = num_c

        # Register each sigma_i as its own parameter with its own constraint
        for i in range(num_c):
            self.register_parameter(
                name=f"raw_n_sum_scale_{i}",
                parameter=torch.nn.Parameter(torch.zeros(()))  # scalar parameter
            )
            self.register_constraint(
                f"raw_n_sum_scale_{i}",
                gpytorch.constraints.Positive()
            )

    @property
    def n_sum_scales(self):
        """
        Returns a tensor of constrained sigma_i of shape (num_c,).
        """
        sigmas = []
        for i in range(self.num_sigmas):
            raw = getattr(self, f"raw_n_sum_scale_{i}")
            constraint = getattr(self, f"raw_n_sum_scale_{i}_constraint")
            sigmas.append(constraint.transform(raw))
        return torch.stack(sigmas)

    def forward(self, x1, x2, diag: bool = False, **params):
        # Base SE covariance
        base_covar = self.base_kernel(x1, x2, diag=diag, **params)

        # Sum over all constrained sigma_i
        scale = self.n_sum_scales.sum()

        # Scale the covariance
        return base_covar.mul(scale)



class NNumCSEKernel(gpytorch.kernels.Kernel):
    """
    k(x, x') = (sum_i sigma_i) * SE(x, x')
    with fixed SE lengthscale = 1.
    """
    def __init__(self, num_c: int = 1, active_dims=None):
        super().__init__(active_dims=active_dims)

        # Base SE (RBF) kernel
        self.base_kernel = gpytorch.kernels.RBFKernel(active_dims=active_dims)

        # Fix lengthscale to 1.0 and disable gradient
        self.base_kernel.lengthscale = 1.0
        self.base_kernel.raw_lengthscale.requires_grad_(False)

        self.num_sigmas = num_c

        # Register each sigma_i as its own parameter with its own constraint
        for i in range(num_c):
            self.register_parameter(
                name=f"raw_n_num_scale_{i}",
                parameter=torch.nn.Parameter(torch.zeros(()))  # scalar parameter
            )
            self.register_constraint(
                f"raw_n_num_scale_{i}",
                gpytorch.constraints.Positive()
            )

    @property
    def n_num_scales(self):
        """
        Returns a tensor of constrained sigma_i of shape (num_c,).
        """
        sigmas = []
        for i in range(self.num_sigmas):
            raw = getattr(self, f"raw_n_num_scale_{i}")
            constraint = getattr(self, f"raw_n_num_scale_{i}_constraint")
            sigmas.append(constraint.transform(raw))
        return torch.stack(sigmas)

    def forward(self, x1, x2, diag: bool = False, **params):
        # Base SE covariance
        base_covar = self.base_kernel(x1, x2, diag=diag, **params)

        # Prod over all constrained sigma_i
        scale = self.n_num_scales.prod()

        # Scale the covariance
        return base_covar.mul(scale)

def _n_sum_c_se(*, active_dims=None, num_c=1, **_):
    return NSumCSEKernel(active_dims=active_dims, num_c=num_c)

def _n_num_c_se(*, active_dims=None, num_c=1, **_):
    return NNumCSEKernel(active_dims=active_dims, num_c=num_c)

class _BaseExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=gpytorch.kernels.RBFKernel,
                 weights=None, active_dims=None):
        super(_BaseExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def neg_loss(train_x, train_y, model, likelihood, parameter_priors, kernel_parameter_priors, model_parameter_prior, MAP=False):
        mll_fkt = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) 
        loss = -mll_fkt(model(train_x), train_y)
        if MAP:
            log_p = log_normalized_prior(model, param_specs=parameter_priors, kernel_param_specs=kernel_parameter_priors, prior=model_parameter_prior)
            loss -= log_p

ALL_METRIC_LOGS = {"prod" : {}, "sum":{}}
order = ["prod", "sum"]
for i, kernel_lambda in enumerate([_n_num_c_se, _n_sum_c_se]):
    for n in list(range(1, 3)):
        # I want to perform a grid search over the parameter space
        # I only want to inspect the 1D and 2D space for both n_sum_c_se and n_num_c_se, i.e. the product and the sum
        # The 1D case is easy, just use 1000 sigma values from -10 to 10
        # The 2D case is a meshgrid with 1000*1000 = 1 million values from (-10, -10) to (10, 10)

        print(f"Run {n}")
        # Initialize GP model with the kernel and the data
        model_likelihood_MLL = gpytorch.likelihoods.GaussianLikelihood()
        #model_MLL = _BaseExactGP(X, y, model_likelihood_MLL, kernel=_n_num_c_se(num_c=n))
        model_MLL = _BaseExactGP(X, y, model_likelihood_MLL, kernel=kernel_lambda(num_c=n))
        model_likelihood_MLL.noise.data = torch.tensor(0.0)
        model_likelihood_MLL.raw_noise.requires_grad = False

        # Define the prior distribution for the model parameters
        model_parameter_prior = prior_distribution(model_MLL, param_specs=None, kernel_param_specs=None, default_mean=0.0, default_std=10.0)
        map = MAP(logarithmic=True, scaling=False)
        mll = MLL(logarithmic=True, scaling=False)

        map_call = lambda : map(model_MLL, model_likelihood_MLL, X, y, prior=model_parameter_prior, logging=True)
        mll_call = lambda : mll(model_MLL, model_likelihood_MLL, X, y, logging=True)

        map_tensor = torch.tensor([])
        mll_tensor = torch.tensor([])
        if n == 1:
            # No grid, just single values
            all_sigmas = torch.linspace(-10.0, 30.0, 1_000)
            for sigma_val in all_sigmas:
                # In place operation
                reparameterize_model_trainable(model_MLL, [sigma_val])
                map_val, map_log = map_call()
                mll_val, mll_log = mll_call()
                if map_tensor.numel() == 0:
                    map_tensor = torch.tensor((sigma_val, map_val))
                    mll_tensor = torch.tensor((sigma_val, mll_val))
                else:
                    map_tensor = torch.vstack((map_tensor, torch.tensor((sigma_val, map_val))))
                    mll_tensor = torch.vstack((mll_tensor, torch.tensor((sigma_val, mll_val))))

        else:
            # Grid!
            sigma_1s = torch.linspace(-10, 30, 500)
            sigma_2s = torch.linspace(-10, 30, 500)
            sigma_mesh = torch.cartesian_prod(sigma_1s, sigma_2s)
            for sigma_val in sigma_mesh:
                reparameterize_model_trainable(model_MLL, sigma_val)
                map_val, map_log = map_call()
                mll_val, mll_log = mll_call()
                if map_tensor.numel() == 0:
                    map_tensor = torch.tensor((*sigma_val, map_val))
                    mll_tensor = torch.tensor((*sigma_val, mll_val))
                else:
                    map_tensor = torch.vstack((map_tensor, torch.tensor((*sigma_val, map_val))))
                    mll_tensor = torch.vstack((mll_tensor, torch.tensor((*sigma_val, mll_val))))
        metric_kut = {n : {"MAP": map_tensor, "MLL":mll_tensor}}

        ALL_METRIC_LOGS[order[i]].update(metric_kut)



    # Posterior plot
    # To plot use the GP prediction and scale everything with the y_shift and y_scale appropriately

# %%
ALL_METRIC_LOGS

# %%
import os
dill.dump(ALL_METRIC_LOGS, open(os.path.join("./", f"diss_example_gridsearch.pkl"), "wb"))
