# %%
from helpers.plotting_functions import plot_3d_gp, plot_3d_data
from helpers.util_functions import prior_distribution
from laplace_model_selection.metrics import Lap0, LapAIC, LapBIC, AIC, BIC, NestedSampling
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import torch

torch.set_default_dtype(torch.float64)
rng_state = torch.tensor(pd.read_csv('rng_state.csv', header=None).values, dtype=torch.uint8).squeeze()
torch.set_rng_state(rng_state)
rng_state = torch.get_rng_state()
rng_state

# %%
import os
import dill
def store(what, where):
    dill.dump(what, open(os.path.join("warping_results", f"{where}.pkl"), "wb"))

# %%
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

# Find all points inside the confidence ellipse
def percentage_inside_ellipse(mu, K, points, sigma_level=2):
    L = np.linalg.cholesky(K)
    threshold = sigma_level ** 2
    count = 0
    for point in points:
        res = np.array(point - mu) @ np.linalg.inv(L)
        if res @ res <= threshold:
            count += 1
    return count / len(points)

# Stolen from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(mu, K, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    cov = K
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0] 

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def nested_sampling_plot(model, model_evidence_log, xdim=0, ydim=1, filter_type="none", std_filter=None, show_last_num=None, return_figure=False, title_add="", fig=None, ax=None, display_figure=True, plot_mll_opt=False, mll_opt_params=None, plot_lap=False, Lap0_logs=None, LapAIC_logs=None, LapBIC_logs=None, lap_colors = ["r", "pink", "black"], Lap_hess=None):

    if not (fig and ax):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    with open(f"{model_evidence_log['res file']}", "rb") as f:
        res = dill.load(f)
    
    # Plot the actual figure
    param_names = [l[0] for l in list(model.named_parameters())]

    # Find the best value and the corresponding hyperparameters
    best_idx = np.argmax(res.logl)
    best_hyperparameters = res.samples[best_idx]

    # Do an outlier cleanup based on the std_filter or the last "show_last_num" samples
    if show_last_num is not None:
        if type(show_last_num) is int:
            # Find value of "show_last_num" sample
            filter_val = sorted(res.logl, reverse=True)[show_last_num]
            mask = res.logl > filter_val
        elif type(show_last_num) is float:
            # Raise an error if show_last_num is not between 0 and 1
            if show_last_num < 0 or show_last_num > 1:
                raise ValueError("show_last_num must be between 0 and 1")
            # assume that it is a percentage of the total samples
            filter_val = sorted(res.logl, reverse=True)[int(len(res.logl)*show_last_num)-1]
            mask = res.logl > filter_val
    # Do an outlier cleanup on res.logz
    elif std_filter is None and not filter_type == "none" and not show_last_num is None:
        raise ValueError("Cannot use both filter_type and show_last_num at the same time")
    
    elif show_last_num is None and not std_filter is None:
        logz_std = np.std(res.logl)
        if filter_type == "max":
            mask = res.logl >= max(res.logl)+std_filter*logz_std
        elif filter_type == "mean":
            raise NotImplementedError("This filter type is not implemented yet")
            logz_mean = np.mean(res.logz)
            mask = np.all(logz_mean - abs(std_filter) * logz_std <= res.logz <= logz_mean + abs(std_filter) * logz_std)
    elif filter_type == "none":
        mask = res.logl == res.logl



    likelihood_surface_scatter = ax.scatter(res.samples[:,xdim][mask], res.samples[:,ydim][mask], c=res.logl[mask], s=3)
    # Best found hyperparameters
    ax.scatter(best_hyperparameters[xdim], best_hyperparameters[ydim], c="r", s=10)

    if plot_mll_opt and not mll_opt_params is None:
        ax.scatter(mll_opt_params[xdim], mll_opt_params[ydim], c="black", s=10)
        # Add a small text beside the point saying "MLL"
        ax.text(mll_opt_params[xdim], mll_opt_params[ydim], "MLL", fontsize=12, color="black", verticalalignment='center', horizontalalignment='right')
    
    coverages = list()
    if plot_lap:
        # Plot the Laplace levels
        for lap_log, lap_color in zip([Lap0_logs, LapAIC_logs, LapBIC_logs], lap_colors):
            if lap_log is None:
                continue
            lap_param_mu = torch.tensor(lap_log["parameter values"])
            # Wait a minute, isn't the Hessian the inverse of the covariance matrix? Yes, see Murphy PML 1 eq. (7.228)
            lap_param_cov_matr = torch.linalg.inv(lap_log["Hessian post correction"])
            # Calculate the amount of samples that are covered by the 1 sigma and 2 sigma interval based on the lap_mu and lap_cov values
            lap_2_sig_coverage = percentage_inside_ellipse(np.array(lap_param_mu).flatten(), lap_param_cov_matr.detach().numpy(), res.samples[mask])
            coverages.append(lap_2_sig_coverage)
            #ax.scatter(lap_param_mu[xdim], lap_param_mu[ydim], c="b", s=10)

            # Plot the std points
            lap_mu_filtered = lap_param_mu.numpy()[[xdim, ydim]] 
            lap_cov_filtered = lap_param_cov_matr.numpy()[[xdim, ydim]][:,[xdim, ydim]]
            #lap_var_ellipse_x, lap_var_ellipse_y = get_std_points(lap_mu_filtered.flatten(), lap_cov_filtered)
            #plt.scatter(lap_var_ellipse_x, lap_var_ellipse_y, c="b", s=1)
            confidence_ellipse(lap_mu_filtered, lap_cov_filtered, ax, n_std=2, edgecolor=lap_color, lw=1)
        if not Lap_hess is None:
            # It can happen that the Hessian is not invertible, in that case drawing the confidence ellipse is not possible
            try:
                lap_param_cov_matr = torch.linalg.inv(Lap_hess)
                lap_2_sig_coverage = percentage_inside_ellipse(lap_param_mu.flatten().numpy(), lap_param_cov_matr.numpy(), res.samples[mask])

                coverages.append(lap_2_sig_coverage)

                lap_mu_filtered = lap_param_mu.numpy()[[xdim, ydim]] 
                lap_cov_filtered = lap_param_cov_matr.numpy()[[xdim, ydim]][:,[xdim, ydim]]
                confidence_ellipse(lap_mu_filtered, lap_cov_filtered, ax, n_std=2, edgecolor="black", lw=1)
            except Exception as e:
                print(e)

    ax.set_title(f"#Accepted samples: {len(res.samples[:,xdim])}\n#Displayed samples: {len(res.samples[:,xdim][mask])}")
    if plot_lap:
        ax.set_title(ax.get_title() + f"; {coverages[0]*100:.0f}% inside 2 sigma")
    if show_last_num is not None:
        ax.set_title(ax.get_title() + f"\n{show_last_num} best accepted samples")
    elif not std_filter is None:
        ax.set_title(ax.get_title() + f"\n{filter_type}: {std_filter:.0e}")
    ax.set_xlabel(param_names[xdim])
    ax.set_ylabel(param_names[ydim])

    plt.colorbar(likelihood_surface_scatter)

    if return_figure:
        return fig, ax
    if display_figure:
        plt.show()
    return None, None


def outlier_cleanup(values, mode="none", filter=None):
    """
    Outlier cleanup function for the nested sampling results.

    Parameters
    ----------
    values : array-like
        The values to be filtered.
    mode : str
        The mode of filtering. Can be "none", "max", "min", "mean", or "value".
    filter : float or int
        The filter value. If mode is "max" or "min", it can be a percentage (float) or a number of samples (int). 
        If mode is "mean", it is the number of standard deviations from the mean. If mode is "value", it is the threshold value.

    """

    # Filter modes: "none", "max", "mean", "value"
    # "none": There is no filtering, just pass everything
    # "max": Only display the highest "std_filter" samples. If "std_filter" is float, use percent, if int, use the number of samples
    # "min": Only display the smallest "std_filter" samples. If "std_filter" is float, use percent, if int, use the number of samples
    # "mean": Only display the samples that are within "std_filter" standard deviations of the mean
    # "value": Only display the samples that are greater than "std_filter"

    masked=None

    if mode == "none":
        masked = values == values
    elif mode == "min":
        if type(filter) is float:
            # Raise an error if show_last_num is not between 0 and 1
            if filter < 0 or filter > 1:
                raise ValueError("filter must be between 0 and 1")
            # assume that it is a percentage of the total samples
            filter_val = sorted(values)[int(len(values)*filter)]
            masked = values < filter_val
        elif type(filter) is int:
            # Find value of "show_last_num" sample
            filter_val = sorted(values)[filter]
            masked = values < filter_val 
    elif mode == "max":
        if type(filter) is float:
            # Raise an error if show_last_num is not between 0 and 1
            if filter < 0 or filter > 1:
                raise ValueError("filter must be between 0 and 1")
            # assume that it is a percentage of the total samples
            filter_val = sorted(values, reverse=True)[int(len(values)*filter)]
            masked = values > filter_val
        elif type(filter) is int:
            # Find value of "show_last_num" sample
            filter_val = sorted(values, reverse=True)[filter]
            masked = values > filter_val 
    elif mode == "mean":
        raise NotImplementedError("This filter type is not implemented yet")
        logz_mean = np.mean(res.logz)
        masked = np.all(logz_mean - abs(filter) * logz_std <= res.logz <= logz_mean + abs(std_filter) * logz_std)
    elif mode == "value":
        masked = values > filter
    return masked



# More general version of the nested sampling plotting function
def scatter_function_visualization(positions, values, fig=None, ax=None, axis_names=None, show_best=False, xdim=0, ydim=1, filter_mode="none", std_filter=None, show_last_num=None, return_figure=False, title_add="", display_figure=True):
   
    if not (fig and ax):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 

    mask = outlier_cleanup(values, filter_mode, std_filter)
    
    filtered_x_dim = positions[:,xdim][mask]
    filtered_y_dim = positions[:,ydim][mask]
    filtered_colors = values[mask]
    likelihood_surface_scatter = ax.scatter(filtered_x_dim, filtered_y_dim, c=filtered_colors, s=3)

    if show_best:
        # Find the best value and the corresponding hyperparameters
        best_idx = np.argmax(values)
        best_hyperparameters = positions[best_idx]
        # Best found hyperparameters
        ax.scatter(best_hyperparameters[xdim], best_hyperparameters[ydim], c="r", s=10)

    if show_last_num is not None:
        ax.set_title(f"#Samples: {len(values)}; {show_last_num} best accepted samples")
    elif not std_filter is None:
        ax.set_title(f"#Samples: {len(values)}; {filter_mode}: {std_filter:.0e}")
    else:
        ax.set_title(f"#Samples: {len(values)}; ")
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])

    plt.colorbar(likelihood_surface_scatter)

    if return_figure:
        return fig, ax
    if display_figure:
        plt.show()
    return None, None


# %%
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np

class TanhWarp(nn.Module):
    """
    g(y) = y + sum_i a_i * tanh(b_i * (y + c_i))
    with a_i, b_i > 0 to guarantee monotonicity: g'(y) = 1 + sum_i a_i*b_i*(1 - tanh(.)^2) > 0
    """
    def __init__(self, M=3, b_transform=None, a_transform=None):
        super().__init__()
        self.raw_a = nn.Parameter(torch.zeros(M))  # -> a = softplus(raw_a)
        self.raw_b = nn.Parameter(torch.zeros(M))  # -> b = softplus(raw_b)
        # The transformation functions for a, b
        # limit b to be in the interval (q, p) resp. for a in (q2, p2)
        #p, q = 10000.0, 0.0
        #self.b_transform = b_transform if b_transform is not None else lambda x: (torch.sigmoid(x) * (p - q)) + q 
        #p2, q2 = 20000.0, 0.0
        #self.a_transform = a_transform if a_transform is not None else lambda x: (torch.sigmoid(x) * (p2 - q2)) + q2
        self.a_transform = torch.nn.Softplus()
        self.b_transform = torch.nn.Softplus()

        self.c     = nn.Parameter(torch.zeros(M))

    def forward(self, y):
        a = self.a_transform(self.raw_a)  # (M,)
        b = self.b_transform(self.raw_b)  # (M,)
        z = y.unsqueeze(-1) + self.c   # (..., M)
        return y + torch.sum(a * torch.tanh(b * z), dim=-1)

    def log_abs_det_jacobian(self, y):
        a = self.a_transform(self.raw_a)
        b = self.b_transform(self.raw_b)
        z = y.unsqueeze(-1) + self.c
        t = torch.tanh(b * z)
        sech2 = 1 - t**2
        deriv = 1.0 + torch.sum(a * b * sech2, dim=-1)
        deriv = torch.clamp(deriv, min=1e-8)  # numerical safety
        return torch.log(deriv)

    @torch.no_grad()
    def inverse(self, u):
        u_scaled = u/self.a_transform(self.raw_a) 
        y = 1/self.b_transform(self.raw_b) * torch.atanh(u_scaled) - self.c
        if torch.any(torch.isnan(y)):
            # replace the nans with the values in u
            y[torch.isnan(y)] = u[torch.isnan(y)]
            for _ in range(40):
                gy  = self.forward(y)
                dgy = torch.exp(self.log_abs_det_jacobian(y))
                y   = y - (gy - u)/dgy
        return y


#    @torch.no_grad()
#    def inverse(self, u, y_init=None, iters=40):
#        # Newton solve for y: g(y)=u ; monotone g ensures unique root
#        y = u.clone() if y_init is None else y_init.clone()
#        for _ in range(iters):
#            gy  = self.forward(y)
#            dgy = torch.exp(self.log_abs_det_jacobian(y))
#            y   = y - (gy - u)/dgy
#        return y
#

# %%
import gpytorch
class Model(gpytorch.models.ExactGP):
    def __init__(self, X, u, likelihood, **kwargs):
        super().__init__(X, u, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.manifold = kwargs.get("manifold", lambda x: x)
        self.warp = kwargs.get("warp", lambda x: x)
        self.y = kwargs.get("y", [])
        self.X = X
        self.old_params = self.parameters()
        self.apply_warp()


    def forward(self, x):
        mean_x = self.mean_module(self.manifold(x))
        covar_x = self.covar_module(self.manifold(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def apply_warp(self):
        if self.old_params == self.parameters():
            return
        self.u = self.warp(self.y)
        self.u_mean = self.u.mean()
        self.u_std = self.u.std()
        self.u = self.u - self.u.mean()  # center the data
        self.u = self.u / self.u.std()   # normalize the data
        self.set_train_data(self.X, self.u)
        self.old_params = self.parameters()

    def inverse_normalize(self, u):
        return u * self.u_std + self.u_mean


class MIModel(gpytorch.models.ExactGP):
    def __init__(self, X, u, likelihood, **kwargs):
        super().__init__(X, u, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        k0 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=0))
        k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=1))
        self.covar_module = k0 * k1
        if "warp" in kwargs:
            self.warp = kwargs.get("warp", lambda x: None)
            self.y = kwargs.get("y", [])
            self.X = X
            self.old_params = self.parameters()
            self.apply_warp()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    def apply_warp(self):
        if self.old_params == self.parameters():
            return
        self.u = self.warp(self.y)
        self.set_train_data(self.X, self.u)
        self.old_params = self.parameters()

# %%
#in 1d:
## Daten von x=0..1
#f:=x->2+exp(exp(3-100*x))-1/5*exp(-1/2*(x-0.1)^2/(1/5)^2)+0.05*x;
#
#Noise:
#heteroscedatic!
#the noise is Gaussian in log-space with a standard deviation of 5%
#[x->exp(0.95*log(f(x))),f,x->exp(1.05*log(f(x)))]
#
## baue 2. Input "a" ein. a=0..1. Hat weniger einfluss
#g:=x->f(x-0.1*sin(5*a/2))*(1+0.1*arctan(5*a-2.5))+0.1*cos(10*a/3);


# Requires limits of a in [0, 4], b in [0, 1]
data_f = lambda x: 2 + torch.exp(torch.exp(3 - 100 * x)) - 1/5 * torch.exp(-1/2 * (x - 0.1)**2 / (1/5)**2) + 0.05 * x
# Requires limits of a in [0, 2], b in [0, 1]
data_noise = lambda x: torch.exp(torch.log(data_f(x))*(1 + 0.05 * torch.randn_like(x)))  # heteroscedastic noise
# Requires limits of a in [0, 1], b in [0, 1]
# Original
#data_g = lambda x, a: data_f(x - 0.1 * torch.sin(5 * a / 2)) * (1 + 0.1 * torch.arctan(5 * a - 2.5)) + 0.1 * torch.cos(10 * a / 3)
data_g = lambda x, a: data_f(x - 0.01 * torch.sin(5 * a / 2) ) * (1 + 0.1 * torch.arctan(5 * a - 2.5)) + 0.1 * torch.cos(10 * a / 3)
data_g2 = lambda x: data_g(x[:, 0], x[:, 1])


X = torch.cat([torch.linspace(0, 0.001, 10), torch.linspace(0.01, 1, 50)])
#X = torch.linspace(0, 1.0, 100)

y = data_noise(X) 
#y = data_g2(coords) 
y = y - y.mean()  # center the data
y = y / y.std()  # normalize the data


# %%
likelihood = gpytorch.likelihoods.GaussianLikelihood()
warp = TanhWarp(M=1)
mani = TanhWarp(M=1)
u = warp(y)
#model = Model(X, u, likelihood, warp=warp, y=y, manifold=mani)
model = Model(X, u, likelihood, warp=warp, y=y)
if "coords" in locals():
    X = coords
#model = MIModel(X, u, likelihood, warp=warp, y=y)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
model.train(); likelihood.train()
optimizer.zero_grad()

# %%
if "coords" in locals() and coords.ndim > 1:
    z_vals = model.u.detach()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], model.u.detach().numpy(),
                c=z_vals.numpy(), cmap='viridis', alpha=0.8)

# %%
if not "coords" in locals():
    plt.plot(model.X.numpy(), model.y.numpy(), label='Original Data', linestyle='dashed')
    plt.plot(model.X.numpy(), model.u.detach().numpy(), label='Warped Data', linestyle='dotted')
    plt.xlim(-0.01, 0.1)
    plt.legend()

# %%
from helpers.training_functions import granso_optimization
from helpers.util_functions import log_normalized_prior, get_full_kernels_in_kernel_expression, randomize_model_hyperparameters
torch.autograd.set_detect_anomaly(True)

kernel_parameter_priors = {
    ("RBFKernel", "lengthscale"): {"mean": 0.0, "std": 10.0}, 
    ("MaternKernel", "lengthscale"): {"mean": 0.0, "std": 10.0},
    ("LinearKernel", "variance"): {"mean": 0.0, "std": 10.0},
    ("AffineKernel", "variance"): {"mean": 0.0, "std": 10.0},
    ("RQKernel", "lengthscale"): {"mean": 0.0, "std": 10.0},
    ("RQKernel", "alpha"): {"mean": 0.0, "std": 10.0},
    ("CosineKernel", "period_length"): {"mean": 0.0, "std": 10.0},
    ("PeriodicKernel", "lengthscale"): {"mean": 0.0, "std": 10.0},
    ("PeriodicKernel", "period_length"): {"mean": 0.0, "std": 10.0},
    ("ScaleKernel", "outputscale"): {"mean": -5.0, "std": 10.0},
    ("LODE_Kernel", "signal_variance_2_0"): {"mean": 0.0, "std": 10.0},  # full match
    ("LODE_Kernel", "lengthscale"): {"mean": 0.0, "std": 10.0},          # base fallback
}


parameter_priors = {
    "likelihood.noise_covar.raw_task_noises": {"mean": 0.1, "std": 0.1},
    "likelihood.noise_covar.raw_noise": {"mean": -3.0, "std": 2.2},
    "warp.raw_a": {"mean": -3.0, "std": 3.0},
    "warp.raw_b": {"mean": -3.0, "std": 3.0},
    "warp.c": {"mean": 0.0, "std": 10.0},
    "manifold.raw_a": {"mean": -3.0, "std": 3.0},
    "manifold.raw_b": {"mean": -3.0, "std": 3.0},
    "manifold.c": {"mean": 0.0, "std": 10.0},
}


kernel_param_specs = {
    ("RBFKernel", "lengthscale"): {"bounds": (1e-0, 1e+1)}, # add ', "type": "uniform"},' # to use uniform distribution
    ("MaternKernel", "lengthscale"): {"bounds": (1e-1, 1.0)},
    ("LinearKernel", "variance"): {"bounds": (1e-1, 1.0)},
    ("AffineKernel", "variance"): {"bounds": (1e-1, 1.0)},
    ("RQKernel", "lengthscale"): {"bounds": (1e-1, 1.0)},
    ("RQKernel", "alpha"): {"bounds": (1e-1, 1.0)},
    ("CosineKernel", "period_length"): {"bounds": (1e-1, 10.0), "type": "uniform"},
    ("PeriodicKernel", "lengthscale"): {"bounds": (1e-1, 5.0)},
    ("PeriodicKernel", "period_length"): {"bounds": (1e-1, 10.0), "type": "uniform"},
    ("ScaleKernel", "outputscale"): {"bounds": (1e-0, 1e+2)},
    #("LODE_Kernel", "signal_variance_2_0"): {"bounds": (0.05, 0.5)},  # full match
    ("LODE_Kernel", "signal_variance"): {"bounds": (1e-1, 10)},  # base
    ("LODE_Kernel", "lengthscale"): {"bounds": (1e-1, 5.0)},           
}


param_specs = {
    "likelihood.noise_covar.raw_task_noises": {"bounds": (1e-0, 1e+2)},
    "likelihood.noise_covar.raw_noise": {"bounds": (1e-0, 1e+2)},
    "warp.raw_a": {"bounds": (1e-0, 3e+1), "type": "uniform"},
    "warp.raw_b": {"bounds": (1e-0, 3e+1), "type": "uniform"},
    "warp.c": {"bounds": (-1e+1, 1e+1), "type": "uniform"},
    "manifold.raw_a": {"bounds": (1e-0, 3e+1), "type": "uniform"},
    "manifold.raw_b": {"bounds": (1e-0, 3e+1), "type": "uniform"},
    "manifold.c": {"bounds": (-1e+1, 1e+1), "type": "uniform"},
}




# %%
from laplace_model_selection.metrics import reparameterize_and_pos_mll
from helpers.util_functions import reparameterize_model_trainable 

def warping_log_like(theta_i):
    try:
        reparameterize_model_trainable(model, theta_i)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        with torch.no_grad():
            model.apply_warp()
            output = model.likelihood(model(model.X))
            # TODO PyGRANSO dying is a severe problem. as it literally exits the program instead of raising an error
            # negative scaled MLL
            return (mll(output, model.u) - model.warp.log_abs_det_jacobian(model.y).sum()/model.y.numel()) * model.y.numel()
    except Exception as E:
        #print(E)
        log_like = -np.inf
    return log_like

# %%
if not "model_evidence_ns" in locals():
    model_parameter_prior = prior_distribution(model, param_specs=parameter_priors, kernel_param_specs=kernel_parameter_priors, default_mean=0.0, default_std=10.0)
    #model_parameter_prior = prior_distribution(model, default_mean=0.0, default_std=10.0)
    nested = NestedSampling(model=model, prior=model_parameter_prior, store_full=True,  maxcall=1e+6, maxiter=1e+6, sampler="rwalk")
    nested.loglike = warping_log_like
    model_evidence_ns, ns_logs = nested(logging=True)
    store(model_evidence_ns, "model_evidence_ns_rwalk")
    store(ns_logs, "ns_logs_rwalk")