import copy
import dynesty
from dynesty import plotting as dyplot
from GaussianProcess import ExactGPModel
from globalParams import options
import gpytorch
from gpytorch.kernels import ScaleKernel
from helpFunctions import get_string_representation_of_kernel as gsr, clean_kernel_expression, print_formatted_hyperparameters
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import re
from scipy.special import lambertw
import stan
import scipy
import sys
import time
import torch
import threading
import os


def prior_distribution(model):
    # params -
    # TODO de-spaghettize this once the priors are coded properly
    prior_dict = {'SE': {'raw_lengthscale' : {"mean": -0.21221139138922668 , "std":1.8895426067756804}},
                  'MAT52': {'raw_lengthscale' :{"mean": 0.7993038925994188, "std":2.145122566357853 } },
                  'MAT32': {'raw_lengthscale' :{"mean": 1.5711054238673443, "std":2.4453761235991216 } },
                  'RQ': {'raw_lengthscale' :{"mean": -0.049841950913676276, "std":1.9426354614713097 },
                          'raw_alpha' :{"mean": 1.882148553921053, "std":3.096431944989054 } },
                  'PER':{'raw_lengthscale':{"mean": 0.7778461197268618, "std":2.288946656544974 },
                          'raw_period_length':{"mean": 0.6485334993738499, "std":0.9930632050553377 } },
                  'LIN':{'raw_variance' :{"mean": -0.8017903983055685, "std":0.9966569921354465 } },
                  'c':{'raw_outputscale':{"mean": -1.6253091096349706, "std":2.2570021716661923 } },
                  'noise': {'raw_noise':{"mean": -3.51640656386717, "std":3.5831320474767407 }}}
    #prior_dict = {"SE": {"raw_lengthscale": {"mean": 0.891, "std": 2.195}},
    #              "MAT": {"raw_lengthscale": {"mean": 1.631, "std": 2.554}},
    #              "PER": {"raw_lengthscale": {"mean": 0.338, "std": 2.636},
    #                      "raw_period_length": {"mean": 0.284, "std": 0.902}},
    #              "LIN": {"raw_variance": {"mean": -1.463, "std": 1.633}},
    #              "c": {"raw_outputscale": {"mean": -2.163, "std": 2.448}},
    #              "noise": {"raw_noise": {"mean": -1.792, "std": 3.266}}}
    total_log_prob = torch.tensor(0.0)
    variances_list = list()
    debug_param_name_list = list()
    theta_mu = list()
    params = None 
    covar_string = gsr(model.covar_module)
    covar_string = covar_string.replace("(", "")
    covar_string = covar_string.replace(")", "")
    covar_string = covar_string.replace(" ", "")
    covar_string = covar_string.replace("PER", "PER+PER")
    covar_string = covar_string.replace("RQ", "RQ+RQ")
    covar_string_list = [s.split("*") for s in covar_string.split("+")]
    covar_string_list.insert(0, ["LIKELIHOOD"])
    covar_string_list = list(chain.from_iterable(covar_string_list))
    both_PER_params = False
    for (param_name, param), cov_str in zip(model.named_parameters(), covar_string_list):
        if params == None:
            params = param
        else:
            if len(param.shape)==0:
                params = torch.cat((params,param.unsqueeze(0)))
            elif len(param.shape)==1:
                params = torch.cat((params,param))
            else:
                params = torch.cat((params,param.squeeze(0)))
        debug_param_name_list.append(param_name)
        curr_mu = None
        curr_var = None
        # First param is (always?) noise and is always with the likelihood
        if "likelihood" in param_name:
            curr_mu = prior_dict["noise"]["raw_noise"]["mean"]
            curr_var = prior_dict["noise"]["raw_noise"]["std"]
        else:
            if (cov_str == "PER" or cov_str == "RQ") and not both_PER_params:
                curr_mu = prior_dict[cov_str][param_name.split(".")[-1]]["mean"]
                curr_var = prior_dict[cov_str][param_name.split(".")[-1]]["std"]
                both_PER_params = True
            elif (cov_str == "PER" or cov_str == "RQ") and both_PER_params:
                curr_mu = prior_dict[cov_str][param_name.split(".")[-1]]["mean"]
                curr_var = prior_dict[cov_str][param_name.split(".")[-1]]["std"]
                both_PER_params = False
            else:
                try:
                    curr_mu = prior_dict[cov_str][param_name.split(".")[-1]]["mean"]
                    curr_var = prior_dict[cov_str][param_name.split(".")[-1]]["std"]
                except Exception as E:
                    import pdb
                    #pdb.set_trace()
                    prev_cov = cov_str
        theta_mu.append(curr_mu)
        variances_list.append(curr_var)
    theta_mu = torch.tensor(theta_mu)
    theta_mu = theta_mu.unsqueeze(0).t()
    sigma = torch.diag(torch.Tensor(variances_list))
    variance = sigma@sigma
    return theta_mu, variance
 
def log_normalized_prior(model, theta_mu=None, sigma=None):
    theta_mu, sigma = prior_distribution(model) if theta_mu is None or sigma is None else (theta_mu, sigma)
    prior = torch.distributions.MultivariateNormal(theta_mu.t(), sigma)

    params = None
    for (param_name, param) in model.named_parameters():
        if params == None:
            params = param
        else:
            if len(param.shape)==0:
                params = torch.cat((params,param.unsqueeze(0)))
            elif len(param.shape)==1:
                params = torch.cat((params,param))
            else:
                params = torch.cat((params,param.squeeze(0)))
 
    # for convention reasons I'm diving by the number of datapoints
    log_prob = prior.log_prob(params) / len(*model.train_inputs)
    return log_prob.squeeze(0)

def calculate_BIC(loss, num_params, num_data):
    start = time.time()
    BIC = +num_params*torch.log(num_data) - 2*loss
    end = time.time()
    logables = {"punish term" : num_params*torch.log(num_data),
                "Total time": end - start,
                "loss term": 2*loss}
    return BIC, logables


def calculate_AIC(loss, num_params):
    start = time.time()
    AIC = 2*num_params - 2*loss
    end = time.time()
    logables = {"punish term" : 2*num_params,
                "Total time": end - start,
                "loss term": 2*loss}
    return AIC, logables


def Eigenvalue_correction(hessian, param_punish_term):
    # Appendix E.2
    vals, vecs = torch.linalg.eigh(hessian)
    constructed_eigvals = torch.diag(torch.Tensor(
        [max(val, (torch.exp(torch.tensor(-2*param_punish_term))*(2*torch.pi))) for i, val in enumerate(vals)]))
    #constructed_eigvals = torch.diag(torch.Tensor(
    #    [max(val, (torch.exp(torch.tensor(2*param_punish_term))*(6.283))) for i, val in enumerate(vals)]))
    num_replaced = torch.count_nonzero(vals - torch.diag(constructed_eigvals))
    corrected_hessian = vecs@constructed_eigvals@vecs.t()
    return corrected_hessian, torch.diag(constructed_eigvals), num_replaced
        

def Eigenvalue_correction_likelihood_laplace(hessian, theta_mu, params, sigma, param_punish_term):
    # Appendix E.4
    vals, vecs = torch.linalg.eigh(hessian)
    #vecs = vecs.real
    theta_bar = vecs@(theta_mu - params)
    sigma_bar = vecs@sigma@vecs.t()

    def cor(i):
        import pdb
        c = (theta_bar[i])**2/sigma_bar[i][i]
        lamw_val = np.real(lambertw(c * torch.exp(c + 2*param_punish_term)))
        return (c/(sigma_bar[i][i]*lamw_val) - 1/sigma_bar[i][i])

        # return -((trans_theta_mu[i] - trans_params[i])**2 /
        #          (np.real(lambertw((trans_theta_mu[i] - trans_params[i])**2
        #                            * 1/trans_sigma[i][i]
        #                            * torch.exp((trans_theta_mu[i] - trans_params[i])**2
        #                                        * 1/trans_sigma[i][i]-param_punish_term)))
        #           * trans_sigma[i][i]**2)+
        #           (1/trans_sigma[i][i])

    constructed_eigvals = torch.diag(torch.Tensor(
        [max(val, cor(i)) for i, val in enumerate(vals)]))
    num_replaced = torch.count_nonzero(vals - torch.diag(constructed_eigvals))
    corrected_hessian = vecs@constructed_eigvals@vecs.t()
    #print(f"new vals: {torch.linalg.eigh(corrected_hessian)[0]}")
    if any(torch.diag(constructed_eigvals) < -1e-10):
        print("Something went horribly wrong with the c(i)s")
        import pdb
        pdb.set_trace()
        print(constructed_eigvals)
    return corrected_hessian, torch.diag(constructed_eigvals), num_replaced


def calculate_laplace(model, loss_of_model, variances_list=None, likelihood_laplace=False, param_punish_term = -1.0, **kwargs):
    torch.set_default_tensor_type(torch.DoubleTensor)
    """
        likelihood_laplace - Decides whether the original Laplace or the 
                             likelihood Laplace approximation is used
        loss_of_model - The positive optimal log likelihood from PyTorch 
    """
    theta_mu = kwargs["theta_mu"] if "theta_mu" in kwargs else None
    logables = {}
    total_start = time.time()
    # Save a list of model parameters and compute the Hessian of the MLL
    params_list = [p for p in model.parameters()]
    # This is now the negative MLL
    mll = -loss_of_model
    start = time.time()
    try:
        env_grads = torch.autograd.grad(mll, params_list, retain_graph=True, create_graph=True, allow_unused=True)
    except Exception as E:
        print(E)
        import pdb
        pdb.set_trace()
    hess_params = []
    # Calcuate -\nabla\nabla log(f(\theta)) (i.e. Hessian of negative log marginal likelihood)
    for i in range(len(env_grads)):
            hess_params.append(torch.autograd.grad(env_grads[i], params_list, retain_graph=True, allow_unused=True))
    end = time.time()
    derivative_calc_time = end - start
    # TODO de-spaghettize this with prior function that generated mu and var
    prior_dict = {'SE': {'raw_lengthscale' : {"mean": -0.21221139138922668 , "std":1.8895426067756804}},
                'MAT52': {'raw_lengthscale' :{"mean": 0.7993038925994188, "std":2.145122566357853 } },
                'MAT32': {'raw_lengthscale' :{"mean": 1.5711054238673443, "std":2.4453761235991216 } },
                'RQ': {'raw_lengthscale' :{"mean": -0.049841950913676276, "std":1.9426354614713097 }, 
                        'raw_alpha' :{"mean": 1.882148553921053, "std":3.096431944989054 } },
                'CosineKernel':{'raw_period_length':{"mean": 0.6485334993738499, "std":0.9930632050553377 }},
                'PER':{'raw_lengthscale':{"mean": 0.7778461197268618, "std":2.288946656544974 },
                        'raw_period_length':{"mean": 0.6485334993738499, "std":0.9930632050553377 } },
                'LIN':{'raw_variance' :{"mean": -0.8017903983055685, "std":0.9966569921354465 } },
                'c':{'raw_outputscale':{"mean": -1.6253091096349706, "std":2.2570021716661923 } },
                'noise': {'raw_noise':{"mean": -3.51640656386717, "std":3.5831320474767407 }},
                'MyPeriodKernel':{'raw_period_length':{"mean": 0.6485334993738499, "std":0.9930632050553377 }}}

    start = time.time()
    if theta_mu is None:
        theta_mu = []
    if variances_list is None:
        variances_list = []
    debug_param_name_list = []
    if variances_list == [] and theta_mu == []:
        covar_string = gsr(model.covar_module)
        covar_string = covar_string.replace("(", "")
        covar_string = covar_string.replace(")", "")
        covar_string = covar_string.replace(" ", "")
        covar_string = covar_string.replace("PER", "PER+PER")
        covar_string = covar_string.replace("RQ", "RQ+RQ")
        covar_string_list = [s.split("*") for s in covar_string.split("+")]
        covar_string_list.insert(0, ["LIKELIHOOD"])
        covar_string_list = list(chain.from_iterable(covar_string_list))
        both_PER_params = False
        for (param_name, param), cov_str in zip(model.named_parameters(), covar_string_list):
            debug_param_name_list.append(param_name)
            # First param is (always?) noise and is always with the likelihood
            if "likelihood" in param_name:
                theta_mu.append(prior_dict["noise"]["raw_noise"]["mean"])
                variances_list.append(prior_dict["noise"]["raw_noise"]["std"])
                continue
            else:
                if (cov_str == "PER" or cov_str == "RQ") and not both_PER_params:
                    theta_mu.append(prior_dict[cov_str][param_name.split(".")[-1]]["mean"])
                    variances_list.append(prior_dict[cov_str][param_name.split(".")[-1]]["std"])
                    both_PER_params = True
                elif (cov_str == "PER" or cov_str == "RQ") and both_PER_params:
                    theta_mu.append(prior_dict[cov_str][param_name.split(".")[-1]]["mean"])
                    variances_list.append(prior_dict[cov_str][param_name.split(".")[-1]]["std"])
                    both_PER_params = False
                else:
                    try:
                        theta_mu.append(prior_dict[cov_str][param_name.split(".")[-1]]["mean"])
                        variances_list.append(prior_dict[cov_str][param_name.split(".")[-1]]["std"])
                    except Exception as E:
                        import pdb
                        pdb.set_trace()
                        prev_cov = cov_str
    theta_mu = torch.tensor(theta_mu)
    theta_mu = theta_mu.unsqueeze(0).t()

    # sigma is a matrix of variance priors
    sigma = torch.diag(torch.Tensor(variances_list))
    sigma = sigma@sigma
    params = torch.tensor(params_list).clone().reshape(-1, 1)

    end = time.time()
    prior_generation_time = end - start

    hessian = torch.tensor(hess_params).clone()
    hessian = (hessian + hessian.t()) / 2
    hessian = hessian.to(torch.float64)

    vals, T = torch.linalg.eigh(hessian)
    oldHessian = hessian.clone()
    if param_punish_term == "BIC":
        param_punish_term = -0.5*torch.log(torch.tensor(model.train_targets.numel()))
    if not likelihood_laplace:
        # Appendix E.1
        # it says mll, but it's actually the MAP here
        start = time.time()
        hessian, constructed_eigvals_log, num_replaced = Eigenvalue_correction(hessian, param_punish_term)
        end = time.time()
        hessian_correction_time = end - start
        # 1.8378 = log(2pi)
        #punish_term = 0.5*len(theta_mu)*torch.tensor(1.8378) + 0.5*torch.log(torch.det(hessian))
        punish_term = 0.5*len(theta_mu)*torch.tensor(1.8378) - 0.5*torch.sum(torch.log(constructed_eigvals_log))
        laplace = loss_of_model + punish_term
        #punish_without_replacement = 0.5*len(theta_mu)*torch.tensor(1.8378) - 0.5*torch.logdet(oldHessian)
        laplace_without_replacement = loss_of_model + 0.5*len(theta_mu)*torch.tensor(1.8378) - 0.5*torch.logdet(oldHessian)
        end = time.time()
        approximation_time = end - start
        if param_punish_term == -1.0:
            if (len(theta_mu) + punish_term) > 1e-4:
                print("Something went horribly wrong with the c(i)s")
                import pdb
                pdb.set_trace()
                print(constructed_eigvals_log)
        else:
            if (len(theta_mu) + punish_term) > len(theta_mu)+1e-4:
                print("Something went horribly wrong with the c(i)s")
                import pdb
                pdb.set_trace()
                print(constructed_eigvals_log)
    else:
        # Appendix E.3
        # Hessian correcting part (for Eigenvalues < 0 < c(i)  )
        start = time.time()
        hessian, constructed_eigvals_log, num_replaced = Eigenvalue_correction_likelihood_laplace(
            hessian, theta_mu, params, sigma, param_punish_term)
        end = time.time()
        #print(f"{num_replaced}")
        hessian_correction_time = end - start

        start = time.time()
        # Here comes what's wrapped in the exp-function:
        thetas_added = params-theta_mu
        thetas_added_transposed = (params-theta_mu).reshape(1, -1)
        middle_term = (sigma.inverse()-hessian).inverse()
        matmuls = thetas_added_transposed @ sigma.inverse() @ middle_term @ hessian @ thetas_added

        # This can probably also be "-0.5 matmuls" where "matmuls" is based on the negative MLL
        punish_term = - (1/2)*torch.log(sigma.det()) - (1/2)*torch.log((sigma.inverse()+hessian).det()) - (1/2) * matmuls
        matmuls_without_replacement = thetas_added_transposed @ sigma.inverse() @ middle_term @ oldHessian @ thetas_added
        punish_without_replacement = - (1/2)*torch.log(sigma.det()) - (1/2)*torch.log((sigma.inverse()+oldHessian).det()) - (1/2) * matmuls_without_replacement
        laplace = loss_of_model + punish_term
        end = time.time()
        approximation_time = end - start

        #print(f"mll - 1/2 log sigma - 1/2 log sigma H + matmuls\n{mll} - {(1/2)*torch.log(sigma.det())} - {(1/2)*torch.log((sigma.inverse()-hessian).det())} + {(1/2) * matmuls}")
        D = torch.diag(constructed_eigvals_log)
        sigma_h = T@(sigma.inverse())@T.t() 
        #print(f"logdet sigma H; matmul\n {0.5*torch.log(torch.linalg.det(sigma_h - D))} ; {0.5*(thetas_added.t()@T.t())@sigma_h@((sigma_h - D).inverse())@D@(T@thetas_added)}")
        if param_punish_term == -1.0:
            if (len(theta_mu) + punish_term) > 1e-4:
                print("Something went horribly wrong with the c(i)s")
                import pdb
                pdb.set_trace()
                print(constructed_eigvals_log)
        else:
            if (len(theta_mu) + punish_term) > len(theta_mu):
                print("Something went horribly wrong with the c(i)s")
                import pdb
                pdb.set_trace()
                print(constructed_eigvals_log)


        #if any(constructed_eigvals_log < 0):
        #    print("Something went horribly wrong with the c(i)s")
        #    import pdb
        #    pdb.set_trace()
        #    print(constructed_eigvals_log)
        #elif punish_term < 0:
        #    print("matmuls positive!!")
        #    import pdb
        #    pdb.set_trace()
        #    print(matmuls)
        #laplace = mll - (1/2)*torch.log(sigma.det()) - (1/2)*torch.log( (sigma.inverse()-hessian).det() )  + (1/2) * matmuls

        #oldLaplace = mll - (1/2)*torch.log(sigma.det()) - (1/2)*torch.log( (sigma.inverse()-oldHessian).det() )  + (1/2) * thetas_added_transposed @ sigma.inverse() @ (sigma.inverse()-oldHessian).inverse() @ oldHessian @ thetas_added
        #print(f"theta_s: {thetas_added_transposed}")
        #print(f"Sigma inv: {sigma.inverse()}")
        #print(f"(sigma.inverse()-hessian): {(sigma.inverse()-hessian)}")
        #print(f"(sigma.inverse()-hessian).inverse(): {(sigma.inverse()-hessian).inverse()}")
        #print(f"Hessian: {hessian}")
        #print(f"Frob. norm(H):{np.linalg.norm(hessian)}")
        #print(f"matmuls: {matmuls}")
        #print(f"----")
        #print(f"param_list:{debug_param_name_list}")
        #print(f"Corrected eig(H):{torch.linalg.eig(hessian)}")
        #print(f"Old  eig(H):{torch.linalg.eig(oldHessian)}")
        #print(f"Symmetry error: {hessian - hessian.t()}")
    


    total_time = end - total_start
    # Everything worth logging
    logables["neg MLL"] = mll 
    logables["punish term"] = punish_term 
    #logables["punish without replacement"] = punish_without_replacement
    logables["laplace without replacement"] = laplace_without_replacement

    logables["num_replaced"] = num_replaced
    logables["parameter list"] = debug_param_name_list
    logables["Jacobian"] = env_grads
    logables["parameter values"] = params
    logables["corrected Hessian"] = hessian
    logables["diag(constructed eigvals)"] = constructed_eigvals_log
    logables["original symmetrized Hessian"] = oldHessian
    logables["prior mean"] = theta_mu
    logables["diag(prior var)"] = torch.diag(sigma)
    logables["likelihood approximation"] = laplace

    # Everything time related
    logables["Derivative time"]         = derivative_calc_time
    logables["Approximation time"]      = approximation_time
    logables["Correction time"]         = hessian_correction_time
    logables["Prior generation time"]   = prior_generation_time
    logables["Total time"]              = total_time

    if not torch.isfinite(laplace) and laplace > 0:
        import pdb
        pdb.set_trace()

    #torch.set_default_tensor_type(torch.FloatTensor)
    return laplace, logables




def generate_STAN_kernel(kernel_representation : str):
    """
    parameter_list : We assume it just contains strings of parameter names
    """
    replacement_dictionary = {
        "c" : "softplus(theta[i])",
        "SE": "gp_exp_quad_cov(x, 1.0, softplus(theta[i]))",
        "MAT52": "gp_matern52_cov(x, 1.0, softplus(theta[i]))",
        "MAT32": "gp_matern32_cov(x, 1.0, softplus(theta[i]))",
        "PER": "gp_periodic_cov(x, 1.0, sqrt(softplus(theta[i])), softplus(theta[i]))",
        "LIN": "softplus(theta[i]) * gp_dot_prod_cov(x, 0.0)",
        "MyPeriodKernel":"gp_periodic_cov(x, 1.0, 1.0, softplus(theta[i]))"
    }
    # Basically do text replacement
    # Take care of theta order!
    # Replace the matrix muliplications by elementwise operation to prevent issues
    kernel_representation = kernel_representation.replace("*", ".*")
    STAN_str_kernel = f"(identity_matrix(dims(x)[1]).*1e-10) + (identity_matrix(dims(x)[1]).*softplus(theta[i])) + {kernel_representation}"
    search_str = "[i]"
    # str.replace(old, new, count) replaces the leftmost entry
    # Thus by iterating over all occurences of search_str I can hack this
    for key in replacement_dictionary:
        STAN_str_kernel = STAN_str_kernel.replace(key, replacement_dictionary[key])
    for i in range(len(re.findall(re.escape(search_str), STAN_str_kernel))):
        STAN_str_kernel = STAN_str_kernel.replace(search_str, f"[{i+1}]", 1)
    return STAN_str_kernel


def generate_STAN_code(kernel_representation : str, lower_bound = -15):
    # Alternative: use 1:dims(v)[0] in the loop
    functions = """
    functions {
        array[] real softplus(array[] real v){
            array[num_elements(v)] real r;
            for (d in 1:num_elements(v)){
                r[d] = log1p(exp(v[d]));
            }
            return r;
        }
        real softplus(real v){
            return log1p(exp(v));
        }
    }
    """


    data = """
    data {
        int N;
        int D;
        array[N] real x;
        vector[N] y;
        vector[D] t_mu;
        matrix[D, D] t_sigma;
    }
    """

    # Old version:
    #vector<lower=-9.2102>[D] theta_tilde;
    # Give it lower bound -3.0 for each parameter to ensure Softplus doesn't reach 0
    parameters = f"""
    parameters {{
        vector<lower={lower_bound}>[D] theta;
    }}
    """
    transformed_parameters = f"""
        transformed parameters {{
            vector[D] theta;
            if(theta_tilde[1] < -30.2102){{
                theta[1] = -30.2102;
            }}else{{
                theta[1] = theta_tilde[1];
            }}
            for(i in 2:D){{
                theta[i] = theta_tilde[i];
            }}
            
        }}
    """
    model = f"""
    model {{
        matrix[N, N] K;
        vector[N] mu;
        theta ~ multi_normal(t_mu, t_sigma);
        K = {generate_STAN_kernel(kernel_representation)};
        mu = zeros_vector(N);
        y ~ multi_normal(mu, K);
    }}
    """

    generated_quantities = f"""
    generated quantities {{
        real lpd;
        matrix[N, N] K;
        vector[N] mu;
        K = {generate_STAN_kernel(kernel_representation)};
        mu = zeros_vector(N);
        lpd = multi_normal_lpdf(y | mu, K);
    }}
    """
    code = functions + data + parameters +model#transformed_parameters+ model #+ generated_quantities
    return code

 
def calculate_mc_STAN(model, likelihood, num_draws, **kwargs):
    logables = dict()
    # Grab variables from kwargs
    log_param_path = kwargs.get("log_param_path", False)    
    log_full_likelihood = kwargs.get("log_full_likelihood", False)
    log_full_posterior = kwargs.get("log_full_posterior", False)
    lower_bound = kwargs.get("lower_bound", -30)
    manual_seed = kwargs.get("manual_seed", None)
    log_kernel_data = kwargs.get("log_kernel_data", False)
    total_start = time.time()
    covar_string = gsr(model.covar_module)

    theta_mu, sigma = prior_distribution(model) 

    STAN_code = generate_STAN_code(covar_string, lower_bound=lower_bound)
    if type(model.train_inputs) == tuple:
        x = model.train_inputs[0].tolist()
        # Assuming I have [[x1], [x2], [x3], ...]
        if not np.ndim(x) == 1:
            x = [t[0] for t in x]
    else:
        x = model.train_inputs.tolist()
    STAN_data = {"N" : len(x),
                 "D" : len(theta_mu),
                 "x" : x,
                 "y" : model.train_targets.tolist(),
                 "t_mu" : [t[0] for t in theta_mu.tolist()],
                 "t_sigma" : sigma.tolist()
    }
    #print("========================")
    #print("data")
    #print(STAN_data)
    #print("Code")
    #print(STAN_code)
    #print("========================")
    start = time.time()
    if manual_seed is None:
        seed = random.randint(0, 1000000)
    else:
        seed = manual_seed
        
    #print(STAN_data)
    #print(STAN_code)
    posterior = stan.build(STAN_code, data=STAN_data, random_seed=seed)
    end = time.time()
    STAN_model_generation_time = end - start
    if num_draws is None:
        raise ValueError("Number of draws not specified")
    start = time.time()

    fit = posterior.sample(num_chains=int(1), num_samples=num_draws)#, num_warmup=int(1))
    end = time.time()
    STAN_MCMC_sampling_time = end - start
    # Use the sampled parameters to reconstruct the mean and cov. matr.
    # Average(?) to get the actual posterior likelihood
    post_frame = fit.to_frame()
    import pdb
    #pdb.set_trace()
    #print(fit)
    manual_lp_list = list()
    manual_post_list = list()
    log_prior_list = list()
    bad_entries = 0
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    def inv_softplus(x):
        return x + torch.log(-torch.expm1(-x))

    start = time.time()
    STAN_like_approx = post_frame["lp__"]
   
    # Iterate over chain
    for row_num, sample in enumerate(post_frame[list(fit.constrained_param_names)].iterrows()):
        # Iterate over kernel parameters
        # Main assumption: Kernel parameters are stored in order of kernel
        # appearance from left to right, just like in STAN
        # Each theta corresponds to exactly one model parameter
        for model_param, sampled_param in zip(model.parameters(), sample[1]):
            #model_param.data = torch.full_like(model_param.data, inv_softplus(torch.tensor(sampled_param)))
            model_param.data = torch.full_like(model_param.data, torch.tensor(sampled_param))


        try:
            #with torch.no_grad():
            #    observed_pred_prior = likelihood(model(model.train_inputs[0]))
            #pdb.set_trace()

            #like_cov_chol = torch.linalg.cholesky(observed_pred_prior.covariance_matrix)
            #like_dist = torch.distributions.multivariate_normal.MultivariateNormal(observed_pred_prior.mean.flatten(), scale_tril=like_cov_chol)
            #manual_lp_list.append(like_dist.log_prob(model.train_targets))
            mll_val = (mll(model(model.train_inputs[0]), model.train_targets)*len(*model.train_inputs)).detach()
            manual_lp_list.append(mll_val)
            manual_post_list.append(mll_val.item() + log_normalized_prior(model).item())

            # TODO write log prior function
            log_prior_list.append(log_normalized_prior(model).item())
        except Exception as e:
            manual_lp_list.append(np.nan)
            bad_entries += 1
            print(e)
            print(list(model.named_parameters()))
            #print(torch.linalg.eig(like_cov_matr.evaluate()))


    parameter_statistics = {
        p: {"mu": np.mean(post_frame[list(fit.constrained_param_names)][p]), 
            "var": np.var(post_frame[list(fit.constrained_param_names)][p])}
                for p in post_frame[list(fit.constrained_param_names)]}

    end = time.time()
    likelihood_approximation_time = end - start
    total_time = end - total_start

    logables["Kernel code"] = STAN_code
    logables["seed"] = seed
    logables["Likelihood time"] = likelihood_approximation_time
    logables["Model compile time"] = STAN_model_generation_time
    logables["Sampling time"] = STAN_MCMC_sampling_time
    logables["Total time"] = total_time
    logables["Bad entries"] = bad_entries
    logables["Parameter statistics"] = parameter_statistics
    logables["Parameter prior"] = {"mu":theta_mu, "var": sigma} 
    logables["likelihood approximation"] = torch.nanmean(torch.Tensor(manual_lp_list))
    logables["posterior approximation"] = torch.nanmean(torch.Tensor(manual_post_list))
    logables["STAN_like_approx"] = STAN_like_approx
    logables["lower_bound"] = lower_bound
    logables["log prior list"] = log_prior_list
    if log_kernel_data:
        logables["Kernel data"] = STAN_data
    if log_full_posterior:
        logables["manual post list"] = manual_post_list
    if log_full_likelihood:
        logables["manual lp list"] = manual_lp_list
    if log_param_path:
        logables["param draws dict"] = post_frame[list(fit.constrained_param_names)]
    ##print(f"Num bad entries: {bad_entries}")
    return torch.nanmean(torch.Tensor(manual_lp_list)), logables









def reparameterize_model(model, theta):
    for model_param, sampled_param in zip(model.parameters(), theta):
        model_param.data = torch.full_like(model_param.data, float(sampled_param))

def reparameterize_and_mll(model, likelihood, theta, train_x, train_y):
    reparameterize_model(model, theta)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    with torch.no_grad():
        mll_val = mll(model(train_x), train_y)
    return mll_val


def NestedSampling(model, **kwargs):
    print_progress = kwargs.get("print_progress", False)
    dynamic_sampling = kwargs.get("dynamic_sampling", True)
    store_samples = kwargs.get("store_samples", False)
    store_likelihoods = kwargs.get("store_likelihoods", False)
    store_full = kwargs.get("store_full", False)
    pickle_directory = kwargs.get("pickle_directory", "")
    checkpoint_file = kwargs.get("checkpoint_file", None)
    maxcall = kwargs.get("maxcall", sys.maxsize)
    checkpoint_every = kwargs.get("checkpoint_every", sys.maxsize)
    res_file_name = kwargs.get("res_file_name", None)
    random_seed = kwargs.get("random_seed", None)
    if random_seed is None:
        random_seed = random.randint(0, 1000000)

    prior_theta_mean, prior_theta_cov = prior_distribution(model)

    # Define the dimensionality of our problem.
    ndim = len(list(model.parameters()))

    def loglike(theta_i):
        try:
            log_like = (reparameterize_and_mll(model, model.likelihood, theta_i, 
                                            model.train_inputs[0], 
                                            model.train_targets)*len(*model.train_inputs)).detach().numpy()
        except Exception as E:
            #print(E)
            log_like = -np.inf
        return log_like

    # Define our prior via the prior transform.
    def prior_transform(u):
        """Transforms the uniform random variables `u ~ Unif[0., 1.)`
        to the parameters of interest."""

        x = np.array(u)  # copy u

        # Bivariate Normal
        t = scipy.stats.norm.ppf(u)  # convert to standard normal
        Csqrt = np.linalg.cholesky(prior_theta_cov.numpy())
        x = np.dot(Csqrt, t)  # correlate with appropriate covariance
        mu = prior_theta_mean.flatten().numpy()  # mean
        x += mu  # add mean
        return x

    rng_generator = np.random.default_rng(seed=random_seed)
    print(f"Random seed: {random_seed}")
    if dynamic_sampling:
        # Trying out dynamic sampler
        dsampler = dynesty.DynamicNestedSampler(loglike, prior_transform, ndim, 
                                                rstate=rng_generator)
        start_time = time.time()
        #dsampler.run_nested(dlogz_init=0.01, maxcall=100000, print_progress=print_progress)# nlive_init=500, nlive_batch=100,
        dsampler.run_nested(dlogz_init=0.01,# nlive_init=500, nlive_batch=100,
                            print_progress=print_progress,
                            maxcall=maxcall)
                            # checkpoint_every=checkpoint_every, # checkpoint_file=checkpoint_file
        end_time = time.time()
        res = dsampler.results

    else:
        # Sample from our distribution.
        sampler = dynesty.NestedSampler(loglike,
                                        prior_transform,
                                        ndim,
                                        bound='multi',
                                        sample='auto',
                                        nlive=500)
        sampler.run_nested(dlogz=0.01, print_progress=print_progress)
        res = sampler.results
    logables = dict()
    logables["Sample time"] = end_time - start_time
    logables["log Z"] = res["logz"][-1]
    logables["log Z err"] = res["logzerr"][-1]
    logables["prior mean"] = prior_theta_mean
    logables["prior cov"] = prior_theta_cov
    logables["dynamic"] = dynamic_sampling
    logables["num sampled"] = res.niter
    logables["parameter statistics"] = {"mu": np.mean(res.samples, axis=0),
                                        "std": np.std(res.samples, axis=0)}
    if store_likelihoods and not store_full:
        logables["log likelihoods"] = res["logl"]
    if store_samples and not store_full:
        logables["samples"] = res["samples"]
    if store_full:
        pickle_filename = f"res_{time.time()}.pkl" if res_file_name is None else res_file_name
        if not os.path.exists(os.path.join(pickle_directory, "Nested_results")):
            os.makedirs(os.path.join(pickle_directory, "Nested_results"))
        full_pickle_path = os.path.join(pickle_directory, "Nested_results", pickle_filename)
        pickle.dump(res, open(full_pickle_path, "wb"))
        logables["res file"] = full_pickle_path
    return res.logz[-1], logables
