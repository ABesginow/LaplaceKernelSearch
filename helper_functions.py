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



hyperparameter_limits = {"RBFKernel": {"lengthscale": [1e-3,5]},
                         "MaternKernel": {"lengthscale": [1e-3,1]},
                         "LinearKernel": {"variance": [1e-4,1]},
                         "AffineKernel": {"variance": [1e-4,1]},
                         "RQKernel": {"lengthscale": [1e-3,1],
                                      "alpha": [1e-3,1]},
                         "CosineKernel": {"period_length": [1e-3,3]},
                         "PeriodicKernel": {"lengthscale": [1e-3,5],
                                            "period_length": [1e-3,10]},
                         "ScaleKernel": {"outputscale": [1e-3,10]},
                         "Noise": [1e-3, 1],
                         "MyPeriodKernel":{"period_length": [1e-3,3]}}

def random_reinit(model, logarithmic=False):
    #print("Random reparameterization")
    #print("old parameters: ", list(model.named_parameters()))
    model_params = model.parameters()
    relevant_hyper_limits = list()
    relevant_hyper_limits.append(hyperparameter_limits["Noise"])
    kernel_name_list = [kernel for kernel in get_full_kernels_in_kernel_expression(model.covar_module)]
    for kernel in kernel_name_list:
        for param_name in hyperparameter_limits[kernel]:
            relevant_hyper_limits.append(hyperparameter_limits[kernel][param_name])

    for i, (param, limit) in enumerate(zip(model_params, relevant_hyper_limits)):
        param_name = limit
        if logarithmic:
            new_param_value = torch.rand_like(param)* (torch.log(torch.tensor(limit[1])) - torch.log(torch.tensor(limit[0]))) + torch.log(torch.tensor(limit[0]))
        else:
            new_param_value = torch.rand_like(param)* (limit[1] - limit[0]) + limit[0]
        param.data = new_param_value
    #print("new parameters: ", list(model.named_parameters()))


def fixed_reinit(model, parameters: torch.tensor) -> None:
    for i, (param, value) in enumerate(zip(model.parameters(), parameters)):
        param.data = torch.full_like(param.data, value)

# Define the training loop
def optimize_hyperparameters(model, likelihood, **kwargs):
    """
    find optimal hyperparameters either by BO or by starting from random initial values multiple times, using an optimizer every time
    and then returning the best result
    """

    # I think this is very ugly to define the class inside the training function and then use a parameter from the function within the class scope. But we all need to make sacrifices...
    # Original class taken from https://ncvx.org/examples/A1_rosenbrock.html
    class HaltLog:
        def __init__(self):
            pass

        def haltLog(self, iteration, x, penaltyfn_parts, d,get_BFGS_state_fn, H_regularized,
                    ls_evals, alpha, n_gradients, stat_vec, stat_val, fallback_level):

            # DON'T CHANGE THIS
            # increment the index/count
            self.index += 1

            # EXAMPLE:
            # store history of x iterates in a preallocated cell array
            self.x_iterates[restart].append(x)
            self.neg_loss[restart].append(penaltyfn_parts.f)
            self.tv[restart].append(penaltyfn_parts.tv)
            self.hessians[restart].append(get_BFGS_state_fn())

            # keep this false unless you want to implement a custom termination
            # condition
            halt = False
            return halt

        # Once PyGRANSO has run, you may call this function to get retreive all
        # the logging data stored in the shared variables, which is populated
        # by haltLog being called on every iteration of PyGRANSO.
        def getLog(self):
            # EXAMPLE
            # return x_iterates, trimmed to correct size
            log = pygransoStruct()
            log.x        = self.x_iterates
            log.neg_loss = self.neg_loss
            log.tv       = self.tv
            log.hessians = self.hessians
            #log = pygransoStruct()
            #log.x   = self.x_iterates[0:self.index]
            #log.f   = self.f[0:self.index]
            #log.tv  = self.tv[0:self.index]
            #log.hessians  = self.hessians[0:self.index]
            return log

        def makeHaltLogFunctions(self, restarts=1):
            # don't change these lambda functions
            halt_log_fn = lambda iteration, x, penaltyfn_parts, d,get_BFGS_state_fn, H_regularized, ls_evals, alpha, n_gradients, stat_vec, stat_val, fallback_level: self.haltLog(iteration, x, penaltyfn_parts, d,get_BFGS_state_fn, H_regularized, ls_evals, alpha, n_gradients, stat_vec, stat_val, fallback_level)

            get_log_fn = lambda : self.getLog()

            # Make your shared variables here to store PyGRANSO history data
            # EXAMPLE - store history of iterates x_0,x_1,...,x_k

            # restart the index and empty the log
            self.index       = 0
            self.x_iterates  = [list() for _ in range(restarts)]
            self.neg_loss    = [list() for _ in range(restarts)]
            self.tv          = [list() for _ in range(restarts)]
            self.hessians    = [list() for _ in range(restarts)]

            # Only modify the body of logIterate(), not its name or arguments.
            # Store whatever data you wish from the current PyGRANSO iteration info,
            # given by the input arguments, into shared variables of
            # makeHaltLogFunctions, so that this data can be retrieved after PyGRANSO
            # has been terminated.
            #
            # DESCRIPTION OF INPUT ARGUMENTS
            #   iter                current iteration number
            #   x                   current iterate x
            #   penaltyfn_parts     struct containing the following
            #       OBJECTIVE AND CONSTRAINTS VALUES
            #       .f              objective value at x
            #       .f_grad         objective gradient at x
            #       .ci             inequality constraint at x
            #       .ci_grad        inequality gradient at x
            #       .ce             equality constraint at x
            #       .ce_grad        equality gradient at x
            #       TOTAL VIOLATION VALUES (inf norm, for determining feasibiliy)
            #       .tvi            total violation of inequality constraints at x
            #       .tve            total violation of equality constraints at x
            #       .tv             total violation of all constraints at x
            #       TOTAL VIOLATION VALUES (one norm, for L1 penalty function)
            #       .tvi_l1         total violation of inequality constraints at x
            #       .tvi_l1_grad    its gradient
            #       .tve_l1         total violation of equality constraints at x
            #       .tve_l1_grad    its gradient
            #       .tv_l1          total violation of all constraints at x
            #       .tv_l1_grad     its gradient
            #       PENALTY FUNCTION VALUES
            #       .p              penalty function value at x
            #       .p_grad         penalty function gradient at x
            #       .mu             current value of the penalty parameter
            #       .feasible_to_tol logical indicating whether x is feasible
            #   d                   search direction
            #   get_BFGS_state_fn   function handle to get the (L)BFGS state data
            #                       FULL MEMORY:
            #                       - returns BFGS inverse Hessian approximation
            #                       LIMITED MEMORY:
            #                       - returns a struct with current L-BFGS state:
            #                           .S          matrix of the BFGS s vectors
            #                           .Y          matrix of the BFGS y vectors
            #                           .rho        row vector of the 1/sty values
            #                           .gamma      H0 scaling factor
            #   H_regularized       regularized version of H
            #                       [] if no regularization was applied to H
            #   fn_evals            number of function evaluations incurred during
            #                       this iteration
            #   alpha               size of accepted size
            #   n_gradients         number of previous gradients used for computing
            #                       the termination QP
            #   stat_vec            stationarity measure vector
            #   stat_val            approximate value of stationarity:
            #                           norm(stat_vec)
            #                       gradients (result of termination QP)
            #   fallback_level      number of strategy needed for a successful step
            #                       to be taken.  See bfgssqpOptionsAdvanced.
            #
            # OUTPUT ARGUMENT
            #   halt                set this to true if you wish optimization to
            #                       be halted at the current iterate.  This can be
            #                       used to create a custom termination condition,
            return [halt_log_fn, get_log_fn]


    random_restarts = kwargs.get("random_restarts", options["training"]["restarts"]+1)
    uninformed = kwargs.get("uninformed", False)
    logarithmic_reinit = kwargs.get("logarithmic_reinit", False)
    train_log = [list() for _ in range(random_restarts)]

    """
    # The call that comes from GRANSO
    user_halt = halt_log_fn(0, x, self.penaltyfn_at_x, np.zeros((n,1)),
                                        get_bfgs_state_fn, H_QP,
                                        1, 0, 1, stat_vec, self.stat_val, 0          )
    """
    def log_fnct(*args):
        train_log[restart].append((args[1], args[2].neg_loss))
        return False 


    train_x = kwargs.get("X", model.train_inputs)
    train_y = kwargs.get("Y", model.train_targets)
    MAP = kwargs.get("MAP", True)
    double_precision = kwargs.get("double_precision", False)

    # Set up the likelihood and model
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #model = GPModel(train_x, train_y, likelihood)

    # Define the negative log likelihood
    mll_fkt = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Set up the PyGRANSO optimizer
    opts = pygransoStruct()
    opts.torch_device = torch.device('cpu')
    nvar = getNvarTorch(model.parameters())
    opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
    opts.opt_tol = float(1e-10)
    #opts.limited_mem_size = int(100)
    opts.limited_mem_size = 0
    opts.globalAD = True
    opts.double_precision = double_precision
    opts.quadprog_info_msg = False
    opts.print_level = int(0)
    opts.halt_on_linesearch_bracket = False
    mHLF_obj = HaltLog()
    [halt_log_fn, get_log_fn] = mHLF_obj.makeHaltLogFunctions(restarts=random_restarts)

    #  Set PyGRANSO's logging function in opts
    opts.halt_log_fn = halt_log_fn

    # Define the objective function
    def objective_function(model):
        output = model(train_x)
        try:
            # TODO PyGRANSO dying is a severe problem. as it literally exits the program instead of raising an error
            # negative scaled MLL
            loss = -mll_fkt(output, train_y)
        except Exception as E:
            print("LOG ERROR: Severe PyGRANSO issue. Loss is inf+0")
            loss = torch.tensor(np.inf, requires_grad=True) + torch.tensor(0)
        if MAP:
            # log_normalized_prior is in metrics.py 
            log_p = log_normalized_prior(model, uninformed=uninformed)
            # negative scaled MAP
            loss -= log_p
        #print(f"LOG: {loss}")
        return [loss, None, None]

    best_model_state_dict = model.state_dict()
    best_likelihood_state_dict = likelihood.state_dict()

    best_f = np.inf
    for restart in range(random_restarts):
        print("---")
        print("start parameters: ", opts.x0)
        # Train the model using PyGRANSO
        try:
            soln = pygranso(var_spec=model, combined_fn=objective_function, user_opts=opts)
            print(f"Restart {restart} : trained parameters: {list(model.named_parameters())}")
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()
            pass

        if soln.final.f < best_f:
            print(f"LOG: Found new best solution: {soln.final.f}")
            best_f = soln.final.f
            best_model_state_dict = model.state_dict()
            best_likelihood_state_dict = likelihood.state_dict()
        random_reinit(model, logarithmic=logarithmic_reinit)
        opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)

    model.load_state_dict(best_model_state_dict)
    likelihood.load_state_dict(best_likelihood_state_dict)
    print(f"----")
    print(f"Final best parameters: {list(model.named_parameters())} w. loss: {best_f} (smaller=better)")
    print(f"----")

    loss = -mll_fkt(model(train_x), train_y)
    #print(f"LOG: Final MLL: {loss}")
    if MAP:
        log_p = log_normalized_prior(model, uninformed=uninformed)
        loss -= log_p
        #print(f"LOG: Final MAP: {loss}")

    #print(f"post training (best): {list(model.named_parameters())} w. loss: {soln.best.f}")
    #print(f"post training (final): {list(model.named_parameters())} w. loss: {soln.final.f}")
    
    #print(torch.autograd.grad(loss, [p for p in model.parameters()], retain_graph=True, create_graph=True, allow_unused=True))
    # Return the trained model
    return loss, model, likelihood, get_log_fn()


def get_std_points(mu, K):
    x, y = np.mgrid[-3:3:.1, -3:3:.1]
    L = np.linalg.cholesky(K)

    data = np.dstack((x, y))

    # Drawing the unit circle
    # x^2 + y^2 = 1
    precision = 50
    unit_x = torch.cat([torch.linspace(-1, 1, precision), torch.linspace(-1, 1, precision)])
    unit_y = torch.cat([torch.sqrt(1 - torch.linspace(-1, 1, precision)**2), -torch.sqrt(1 - torch.linspace(-1, 1, precision)**2)])

    new_unit_x = list()
    new_unit_y = list()

    for tx, ty in zip(unit_x, unit_y):
        res = np.array([tx, ty]) @ L
        new_unit_x.append(mu[0] + 2*res[0])
        new_unit_y.append(mu[1] + 2*res[1])
    return new_unit_x, new_unit_y


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


def log_dill(data, filename):
    with open(filename, 'wb') as f:
        dill.dump(data, f)


