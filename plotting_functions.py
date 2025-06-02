import sys
sys.path.insert(1, "..")
from collections import namedtuple
import dill
from helpFunctions import get_string_representation_of_kernel as gsr
from helpFunctions import get_full_kernels_in_kernel_expression
from globalParams import options
import gpytorch
import itertools
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from metrics import calculate_AIC as AIC, calculate_BIC as BIC, calculate_laplace as Laplace, NestedSampling as Nested, log_normalized_prior, prior_distribution
import numpy as np
import os
import pandas as pd
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import re
import torch

from helper_functions import percentage_inside_ellipse, get_std_points


def plot_parameter_progression(parameter_progression, losses=None, xlabel=None, ylabel=None, fig=None, ax=None, xdim=0, ydim=1, display_figure=True, return_figure=False, title_add=""):
    if not (fig and ax):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Reverse colormap by using "viridis_r"
    cmap = plt.cm.viridis_r  

    # Define a normalization based on loss values
    norm = None
    if losses is not None:
        norm = mcolors.Normalize(vmin=min(losses[1:]), vmax=max(losses[1:]))  # Keep vmin and vmax as usual
    
    for i in range(1, len(parameter_progression)):
        if losses is None:
            colormap = cmap(i / len(parameter_progression))  # Use reversed colormap
        else:
            colormap = cmap(norm(losses[i]))  # Normalize and apply reversed colormap
        
        ax.plot([parameter_progression[i-1][xdim], parameter_progression[i][xdim]], 
                [parameter_progression[i-1][ydim], parameter_progression[i][ydim]], 
                color=colormap)

        ax.annotate("", xy=(parameter_progression[i][xdim], parameter_progression[i][ydim]), 
                    xytext=(parameter_progression[i-1][xdim], parameter_progression[i-1][ydim]),  
                    arrowprops=dict(arrowstyle="->", color=colormap, lw=2))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Parameter progression {title_add}")

    # Add colorbar with reversed colormap
    if losses is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # Use reversed colormap here
        sm.set_array([])  # Required for colorbar
        fig.colorbar(sm, ax=ax, label="Loss value (higher = worse)")

    if return_figure:
        return fig, ax
    if display_figure:
        plt.show()
    return None, None



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
            filter_val = sorted(res.logl, reverse=True)[int(len(res.logl)*show_last_num)]
            mask = res.logl > filter_val

    # Do an outlier cleanup on res.logz
    if std_filter is None and not filter_type == "none" and not show_last_num is None:
        raise ValueError("Cannot use both filter_type and show_last_num at the same time")
    if show_last_num is None and not std_filter is None:
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
            lap_param_mu = lap_log["parameter values"]
            # Wait a minute, isn't the Hessian the inverse of the covariance matrix? Yes, see Murphy PML 1 eq. (7.228)
            lap_param_cov_matr = torch.linalg.inv(lap_log["Hessian post correction"])
            # Calculate the amount of samples that are covered by the 1 sigma and 2 sigma interval based on the lap_mu and lap_cov values
            lap_2_sig_coverage = percentage_inside_ellipse(lap_param_mu.flatten().numpy(), lap_param_cov_matr.numpy(), res.samples[mask])
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


    if show_last_num is not None:
        ax.set_title(f"#Samples: {sum(res.ncall)}; {coverages[0]*100:.0f}% inside 2 sigma\n{show_last_num} best accepted samples")
    elif not std_filter is None:
        ax.set_title(f"#Samples: {sum(res.ncall)}; {coverages[0]*100:.0f}% inside 2 sigma\n{filter_type}: {std_filter:.0e}")
    else:
        ax.set_title(f"#Samples: {sum(res.ncall)}; {coverages[0]*100:.0f}% inside 2 sigma\n")
    ax.set_xlabel(param_names[xdim])

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


def plot_data(X, Y, return_figure=False, title_add="", figure=None, ax=None, display_figure=True):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(X.numpy(), Y.numpy(), 'k.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"Data {title_add}")
    if not return_figure and display_figure:
        plt.show()
    else:
        return fig, ax

def plot_model(model, likelihood, X, Y, display_figure=True, return_figure=False, figure=None,
               ax=None, loss_val=None, loss_type = None):
    interval_length = torch.max(X) - torch.min(X)
    shift = interval_length * options["plotting"]["border_ratio"]
    test_x = torch.linspace(torch.min(
        X) - shift, torch.max(X) + shift, options["plotting"]["sample_points"])

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    with torch.no_grad():
        if not (figure and ax):
            figure, ax = plt.subplots(1, 1, figsize=(8, 6))

        lower, upper = observed_pred.confidence_region()
        ax.plot(X.numpy(), Y.numpy(), 'k.', zorder=2)
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), color="b", zorder=3)
        amount_of_gradient_steps = 30
        alpha_min = 0.05
        alpha_max = 0.8
        alpha = (alpha_max-alpha_min)/amount_of_gradient_steps
        c = ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(
        ), alpha=alpha+alpha_min, zorder=1).get_facecolor()

        for i in range(1, amount_of_gradient_steps):
            ax.fill_between(test_x.numpy(), (lower+(i/amount_of_gradient_steps)*(upper-lower)).numpy(),
                            (upper-(i/amount_of_gradient_steps)*(upper-lower)).numpy(), alpha=alpha, color=c, zorder=1)
        if options["plotting"]["legend"]:
            ax.plot([], [], 'k.', label="Data")
            ax.plot([], [], 'b', label="Mean")
            ax.plot([], [], color=c, alpha=1.0, label="Confidence")
            ax.legend(loc="upper left")
        ax.set_xlabel("Normalized Input")
        ax.set_ylabel("Normalized Output")
        ax.set_title(f"{loss_type}: {loss_val:.2f}")
    if not return_figure and display_figure:
        plt.show()
    else:
        return figure, ax
# make a nested sampling plot, but multiply each point with the respective prior likelihood, thus generating the Posterior surface instead of the likelihood surface


def posterior_surface_plot(model, model_evidence_log, xdim=0, ydim=1, filter_type="none", std_filter=None, show_last_num=None, return_figure=False, title_add="", fig=None, ax=None, display_figure=True, plot_mll_opt=False, mll_opt_params=None, plot_lap=False, Lap0_logs=None, LapAIC_logs=None, LapBIC_logs=None, lap_colors = ["r", "pink", "black"], Lap_hess=None, uninformed=False):

    if not (fig and ax):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    with open(f"{model_evidence_log['res file']}", "rb") as f:
        res = dill.load(f)
    
    # Plot the actual figure
    param_names = [l[0] for l in list(model.named_parameters())]


    # rescale the "res.samples" by the prior likelihood
    prior_mu, prior_sigma = prior_distribution(model=model, uninformed=uninformed)
    prior = torch.distributions.MultivariateNormal(prior_mu.t(), prior_sigma)
    posterior_samples = np.array([logl+prior.log_prob(torch.tensor(sample)).item() for sample, logl in zip(res.samples, res.logl)])

    # Find the best value and the corresponding hyperparameters
    best_idx = np.argmax(posterior_samples)
    best_hyperparameters = res.samples[best_idx]

    # Do an outlier cleanup based on the std_filter or the last "show_last_num" samples
    if show_last_num is not None:
        if type(show_last_num) is int:
            # Find value of "show_last_num" sample
            filter_val = sorted(posterior_samples, reverse=True)[show_last_num]
            mask = posterior_samples > filter_val
        elif type(show_last_num) is float:
            # Raise an error if show_last_num is not between 0 and 1
            if show_last_num < 0 or show_last_num > 1:
                raise ValueError("show_last_num must be between 0 and 1")
            # assume that it is a percentage of the total samples
            filter_val = sorted(posterior_samples, reverse=True)[int(len(posterior_samples)*show_last_num)]
            mask = posterior_samples > filter_val

    # Do an outlier cleanup on res.logz
    if std_filter is None and not filter_type == "none" and not show_last_num is None:
        raise ValueError("Cannot use both filter_type and show_last_num at the same time")
    if show_last_num is None and not std_filter is None:
        logz_std = np.std(posterior_samples)
        if filter_type == "max":
            mask = posterior_samples >= max(posterior_samples)+std_filter*logz_std
        elif filter_type == "mean":
            raise NotImplementedError("This filter type is not implemented yet")
            logz_mean = np.mean(res.logz)
            mask = np.all(logz_mean - abs(std_filter) * logz_std <= res.logz <= logz_mean + abs(std_filter) * logz_std)
        elif filter_type == "none":
            mask = posterior_samples == posterior_samples


    posterior_surface_scatter = ax.scatter(res.samples[:,xdim][mask], res.samples[:,ydim][mask], c=posterior_samples[mask], s=3)
    # Best found hyperparameters
    ax.scatter(best_hyperparameters[xdim], best_hyperparameters[ydim], c="r", s=10)

    if plot_mll_opt and not mll_opt_params is None:
        ax.scatter(mll_opt_params[xdim], mll_opt_params[ydim], c="black", s=10)
        # Add a small text beside the point saying "MLL"
        ax.text(mll_opt_params[xdim], mll_opt_params[ydim], "MAP", fontsize=12, color="black", verticalalignment='center', horizontalalignment='right')
    
    coverages = list()
    if plot_lap:
        # Plot the Laplace levels
        for lap_log, lap_color in zip([Lap0_logs, LapAIC_logs, LapBIC_logs], lap_colors):
            if lap_log is None:
                continue
            lap_param_mu = lap_log["parameter values"]
            # Wait a minute, isn't the Hessian the inverse of the covariance matrix? Yes, see Murphy PML 1 eq. (7.228)
            lap_param_cov_matr = torch.linalg.inv(lap_log["Hessian post correction"])
            # Calculate the amount of samples that are covered by the 1 sigma and 2 sigma interval based on the lap_mu and lap_cov values
            lap_2_sig_coverage = percentage_inside_ellipse(lap_param_mu.flatten().numpy(), lap_param_cov_matr.numpy(), res.samples[mask])
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


    if show_last_num is not None:
        ax.set_title(f"#Samples: {sum(res.ncall)}; {coverages[0]*100:.0f}% inside 2 sigma\n{show_last_num} best accepted samples")
    elif not std_filter is None:
        ax.set_title(f"#Samples: {sum(res.ncall)}; {coverages[0]*100:.0f}% inside 2 sigma\n{filter_type}: {std_filter:.0e}")
    else:
        ax.set_title(f"#Samples: {sum(res.ncall)}; {coverages[0]*100:.0f}% inside 2 sigma\n")
    ax.set_xlabel(param_names[xdim])

    plt.colorbar(posterior_surface_scatter)

    if return_figure:
        return fig, ax
    if display_figure:
        plt.show()
    return None, None


# =============================================
# 3D plotting
# =============================================

def plot_3d_data(samples, xx, yy, return_figure=False, fig=None, ax=None, display_figure=True, title_add = "", shadow=True):
    """
    Similar to plot_3d_gp_samples, but color-codes each (xx, yy) point in 3D.
    'samples' can be a single 1D tensor or multiple samples in a 2D tensor.
    """
    if not (fig and ax):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    #if samples.ndim == 1:
    #    samples = samples.unsqueeze(0)

    #z_vals = samples.reshape(xx.shape)
    z_vals = samples
    ax.scatter(xx.numpy(), yy.numpy(), z_vals.numpy(),
                c=z_vals.numpy(), cmap='viridis', alpha=0.8)


    if shadow:
        # Plot shadows (projection on X-Y plane at z=0)
        ax.scatter(xx.numpy(), yy.numpy(), 
                np.ones_like(z_vals)*min(z_vals.numpy()), 
                c='gray', alpha=0.3, marker='o')



    ax.set_title(f'Data {title_add}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Output Value')
    if not return_figure and display_figure:
        plt.show()
    else:
        return fig, ax


def plot_3d_gp_samples(samples, xx, yy, return_figure=False, fig=None, ax=None, display_figure=True):
    """
    Visualize multiple samples drawn from a 2D-input (xx, yy) -> 1D-output GP in 3D.
    Each sample in 'samples' should be a 1D tensor that can be reshaped to match xx, yy.
    """
    if not (fig and ax):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    for i, sample in enumerate(samples):
        z_vals = sample.reshape(xx.shape)
        ax.plot_surface(xx.numpy(), yy.numpy(), z_vals.numpy(), alpha=0.4)

    ax.set_title('GP Samples in 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Output')
    if not return_figure and display_figure:
        plt.show()
    else:
        return fig, ax


def plot_3d_gp(model, likelihood, data=None, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                resolution=50, return_figure=False, fig=None, ax=None, 
                display_figure=True, loss_val=None, loss_type=None, shadow=False,
                title_add = ""):
    if not (fig and ax):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')


    if data is not None:
        if data.ndim == 1:
            data = data.unsqueeze(0)

        for sample in data:
            xx = sample[0]
            yy = sample[1]
            zz = sample[2]
            ax.scatter(xx.numpy(), yy.numpy(), zz.numpy(), color="red", alpha=0.8)

            if shadow:
                # Plot shadows (projection on X-Y plane at z=0)
                ax.scatter(xx.numpy(), yy.numpy(), 
                        min(data[:,2].numpy()), 
                        c='gray', alpha=0.3, marker='o')
    model.eval()
    likelihood.eval()

    x_vals = torch.linspace(x_min, x_max, resolution)
    y_vals = torch.linspace(y_min, y_max, resolution)
    xx, yy = torch.meshgrid(x_vals, y_vals)
    test_x = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    with torch.no_grad():
        preds = likelihood(model(test_x))
        mean = preds.mean.reshape(resolution, resolution)
        lower, upper = preds.confidence_region()
        lower = lower.reshape(resolution, resolution)
        upper = upper.reshape(resolution, resolution)


    # Plot mean surface
    ax.plot_surface(xx.numpy(), yy.numpy(), mean.numpy(), cmap='viridis', alpha=0.8)

    # Plot lower and upper surfaces
    ax.plot_surface(xx.numpy(), yy.numpy(), lower.numpy(), color='gray', alpha=0.2)
    ax.plot_surface(xx.numpy(), yy.numpy(), upper.numpy(), color='gray', alpha=0.2)

    ax.set_title(f'2D GP in 3D {title_add}')
    if loss_val is not None:
        ax.set_title(f"{ax.title.get_text()}; {loss_type}: {loss_val}")
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Mean and Variance Range')

    if not return_figure and display_figure:
        plt.show()
    else:
        return fig, ax


