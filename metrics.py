import copy
from GaussianProcess import ExactGPModel
from globalParams import options
import gpytorch as gpt
from gpytorch.kernels import ScaleKernel
from helpFunctions import get_string_representation_of_kernel as gsr, clean_kernel_expression, print_formatted_hyperparameters
from helpFunctions import amount_of_base_kernels, get_kernels_in_kernel_expression
from itertools import chain
import numpy as np
import random
import re
from scipy.special import lambertw
import stan
import time
import torch
import threading


def calculate_AIC(loss, num_params):
    logables = {"correction term" : 2*num_params,
                "loss term": 2*loss}
    return -2*num_params + 2*loss, logables


def Eigenvalue_correction(hessian, theta_mu, params, sigma, param_punish_term):
    vals, vecs = torch.linalg.eigh(hessian)
    #vecs = vecs.real
    theta_bar = vecs@(theta_mu - params)
    sigma_bar = vecs@sigma@vecs.t()

    def cor(i):
        import pdb
        c = (theta_bar[i])**2
        lamw_val = np.real(lambertw(c/sigma_bar[i][i] * torch.exp(c/sigma_bar[i][i] - param_punish_term)))
        return -(c/(sigma_bar[i][i]**2*lamw_val) + 1/sigma_bar[i][i])

        # return -((trans_theta_mu[i] - trans_params[i])**2 /
        #          (np.real(lambertw((trans_theta_mu[i] - trans_params[i])**2
        #                            * 1/trans_sigma[i][i]
        #                            * torch.exp((trans_theta_mu[i] - trans_params[i])**2
        #                                        * 1/trans_sigma[i][i]-param_punish_term)))
        #           * trans_sigma[i][i]**2)+
        #           (1/trans_sigma[i][i])

    constructed_eigvals = torch.diag(torch.Tensor(
        [min(val, cor(i)) for i, val in enumerate(vals)]))
    corrected_hessian = vecs@constructed_eigvals@vecs.t()
    #print(f"new vals: {torch.linalg.eigh(corrected_hessian)[0]}")
    if any(torch.diag(constructed_eigvals) > 0):
        print("Something went horribly wrong with the c(i)s")
        import pdb
        pdb.set_trace()
        print(constructed_eigvals)
    return corrected_hessian, torch.diag(constructed_eigvals)

         

def calculate_laplace(model, loss_of_model, variances_list=None, with_prior=False, param_punish_term = 2.0):
    torch.set_default_tensor_type(torch.DoubleTensor)
    """
        with_prior - Decides whether the version of the Laplace approx WITH the
                     prior is used or the one where the prior is not part of
                     the approx.
    """
    logables = {}
    total_start = time.time()
    num_of_observations = len(*model.train_inputs)
    # Save a list of model parameters and compute the Hessian of the MLL
    params_list = [p for p in model.parameters()]
    # This is now the positive MLL
    mll         = (num_of_observations * (-loss_of_model))
    # This is NEGATIVE MLL
    #mll         = (num_of_observations * (loss_of_model))
    start = time.time()
    env_grads   = torch.autograd.grad(mll, params_list, retain_graph=True, create_graph=True)
    hess_params = []
    for i in range(len(env_grads)):
            hess_params.append(torch.autograd.grad(env_grads[i], params_list, retain_graph=True))
    end = time.time()
    derivative_calc_time = end - start
    #prior_dict_softplussed = {"SE": {"lengthscale" : {"mean": 1.607, "std":1.650}},
    #                          "PER":{"lengthscale": {"mean": 1.473, "std":1.582}, "period_length":{"mean": 0.920, "std":0.690}},
    #                          "LIN":{"variance" : {"mean":0.374, "std":0.309}},
    #                          "C":{"outputscale": {"mean":0.427, "std":0.754}},
    #                          "noise": {"noise": {"mean":0.531, "std":0.384}}}

    prior_dict = {"SE": {"raw_lengthscale": {"mean": 0.891, "std": 2.195}},
                  "PER": {"raw_lengthscale": {"mean": 0.338, "std": 2.636}, 
                          "raw_period_length": {"mean": 0.284, "std": 0.902}},
                  "LIN": {"raw_variance": {"mean": -1.463, "std": 1.633}},
                  "c": {"raw_outputscale": {"mean": -2.163, "std": 2.448}},
                  "noise": {"raw_noise": {"mean": -1.792, "std": 3.266}}}

    start = time.time()
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
                if cov_str == "PER" and not both_PER_params:
                    theta_mu.append(prior_dict[cov_str][param_name.split(".")[-1]]["mean"])
                    variances_list.append(prior_dict[cov_str][param_name.split(".")[-1]]["std"])
                    both_PER_params = True
                elif cov_str == "PER" and both_PER_params:
                    theta_mu.append(prior_dict[cov_str][param_name.split(".")[-1]]["mean"])
                    variances_list.append(prior_dict[cov_str][param_name.split(".")[-1]]["std"])
                    both_PER_params = False
                else:
                    try:
                        theta_mu.append(prior_dict[cov_str][param_name.split(".")[-1]]["mean"])
                        variances_list.append(prior_dict[cov_str][param_name.split(".")[-1]]["std"])
                    except:
                        import pdb
                        pdb.set_trace()
            prev_cov = cov_str
        theta_mu = torch.tensor(theta_mu)
        theta_mu = theta_mu.unsqueeze(0).t()

    # sigma is a matrix of variance priors
    sigma = torch.diag(torch.Tensor(variances_list))
    params = torch.tensor(params_list).clone().reshape(-1,1)

    end = time.time()
    prior_generation_time = end - start

    if with_prior:
        #This is the original
        raise NotImplementedError("Not yet done")
        laplace = mll - (1/2)*torch.log(sigma.det()) - (1/2)*(params-theta_mu).t()@sigma.inverse()@(params-theta_mu) - (1/2)*torch.log((-hessian).det())
    else:

        hessian = torch.tensor(hess_params).clone()
        hessian = (hessian + hessian.t()) / 2
        hessian = hessian.to(torch.float64)

        _, T = torch.linalg.eigh(hessian)
        oldHessian = hessian.clone()

        # Hessian correcting part (for Eigenvalues < 0 < c(i)  )
        start = time.time()
        hessian, constructed_eigvals_log = Eigenvalue_correction(
            hessian, theta_mu, params, sigma, param_punish_term)
        end = time.time()
        hessian_correction_time = end - start

        start = time.time()
        # Here comes what's wrapped in the exp-function:
        thetas_added = params-theta_mu
        thetas_added_transposed = (params-theta_mu).reshape(1,-1)
        middle_term = (sigma.inverse()-hessian).inverse()
        matmuls = thetas_added_transposed @ sigma.inverse() @ middle_term @ hessian @ thetas_added

        laplace = mll - (1/2)*torch.log(sigma.det()) - (1/2)*torch.log((sigma.inverse()-hessian).det()) + (1/2) * matmuls
        end = time.time()
        approximation_time = end - start

        total_time = end - total_start

        #oldLaplace = mll - (1/2)*torch.log(sigma.det()) - (1/2)*torch.log( (sigma.inverse()-oldHessian).det() )  + (1/2) * thetas_added_transposed @ sigma.inverse() @ (sigma.inverse()-oldHessian).inverse() @ oldHessian @ thetas_added
        debug=True
        if(debug):
            print(f"params: {params}")
            print(f"theta_mu: {theta_mu}")
            print(f"theta_s: {thetas_added_transposed}")
            print(f"Sigma inv: {sigma.inverse()}")
            print(f"(sigma.inverse()-hessian): {(sigma.inverse()-hessian)}")
            print(f"(sigma.inverse()-hessian).inverse(): {(sigma.inverse()-hessian).inverse()}")
            print(f"Hessian: {hessian}")
            print(f"Frob. norm(H):{np.linalg.norm(hessian)}")
            print(f"matmuls: {matmuls}")
            print(f"----")
            print(f"param_list:{debug_param_name_list}")
            print(f"Corrected eig(H):{torch.linalg.eigh(hessian)}")
            print(f"Old  eig(H):{torch.linalg.eigh(oldHessian)}")
            print(f"Symmetry error: {hessian - hessian.t()}")
            print(f"Constructed eigvals: {constructed_eigvals_log}")
            if matmuls > 0:
                print("matmuls positive!!")
                import pdb
                pdb.set_trace()
                print(matmuls)

        print(f"mll - 1/2 log sigma - 1/2 log sigma H + matmuls\n{mll} - {(1/2)*torch.log(sigma.det())} - {(1/2)*torch.log((sigma.inverse()-hessian).det())} + {(1/2) * matmuls}")
        D = torch.diag(constructed_eigvals_log)
        sigma_h = T@(sigma.inverse())@T.t() 
        print(f"logdet sigma H; matmul\n {0.5*torch.log(torch.linalg.det(sigma_h - D))} ; {0.5*(thetas_added.t()@T.t())@sigma_h@((sigma_h - D).inverse())@D@(T@thetas_added)}")
        #laplace = mll - (1/2)*torch.log(sigma.det()) - (1/2)*torch.log( (sigma.inverse()-hessian).det() )  + (1/2) * matmuls



    # Everything worth logging
    logables["MLL"] = mll
    logables["parameter list"] = debug_param_name_list
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

    torch.set_default_tensor_type(torch.FloatTensor)
    return laplace, logables




def generate_STAN_kernel(kernel_representation : str, parameter_list : list, covar_string_list : list):
    """
    parameter_list : We assume it just contains strings of parameter names
    """
    replacement_dictionary = {
        "c" : "softplus(theta[i])",
        "SE": "gp_exp_quad_cov(x, 1.0, softplus(theta[i]))",
        "PER": "gp_periodic_cov(x, 1.0, sqrt(softplus(theta[i])), softplus(theta[i]))",
        "LIN": "softplus(theta[i]) * gp_dot_prod_cov(x, 0.0)"
    }
    # Basically do text replacement
    # Take care of theta order!
    # Replace the matrix muliplications by elementwise operation to prevent issues
    kernel_representation = kernel_representation.replace("*", ".*")
    STAN_str_kernel = f"identity_matrix(dims(x)[1])*softplus(theta[i]) + {kernel_representation}"
    search_str = "[i]"
    # str.replace(old, new, count) replaces the leftmost entry
    # Thus by iterating over all occurences of search_str I can hack this
    for key in replacement_dictionary:
        STAN_str_kernel = STAN_str_kernel.replace(key, replacement_dictionary[key])
    for i in range(len(re.findall(re.escape(search_str), STAN_str_kernel))):
        STAN_str_kernel = STAN_str_kernel.replace(search_str, f"[{i+1}]", 1)
    return STAN_str_kernel


def generate_STAN_code(kernel_representation : str,  parameter_list : list, covar_string_list : list):
    # Alternative: use 1:dims(v)[0] in the loop
    functions = """
    functions {
        array[] real softplus(array[] real v){
            array[num_elements(v)] real r;
            for (d in 1:num_elements(v)){
                r[d] = log(1.0 + exp(v[d]));
            }
            return r;
        }
        real softplus(real v){
            return log(1.0 + exp(v));
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

    # Give it lower bound -3.0 for each parameter to ensure Softplus doesn't reach 0
    parameters = """
    parameters {
        vector<lower=-3.0>[D] theta;
    }
    """

    model = f"""
    model {{
        matrix[N, N] K;
        vector[N] mu;
        theta ~ multi_normal(t_mu, t_sigma);
        K = {generate_STAN_kernel(kernel_representation, parameter_list, covar_string_list)};
        mu = zeros_vector(N);
        y ~ multi_normal(mu, K);
    }}
    """

    generated_quantities = f"""
    generated quantities {{
        real lpd;
        matrix[N, N] K;
        vector[N] mu;
        K = {generate_STAN_kernel(kernel_representation, parameter_list, covar_string_list)};
        mu = zeros_vector(N);
        lpd = multi_normal_lpdf(y | mu, K);
    }}
    """
    code = functions + data + parameters + model #+ generated_quantities
    return code


def calculate_mc_STAN(model, likelihood, num_draws):
    # Yes, this is code duplication from above.
    # No, I am not happy with this
    prior_dict = {"SE": {"raw_lengthscale" : {"mean": 0.891, "std":2.195}},
                  "PER":{"raw_lengthscale": {"mean": 0.338, "std":2.636}, "raw_period_length":{"mean": 0.284, "std":0.902}},
                  "LIN":{"raw_variance" : {"mean":-1.463, "std":1.633}},
                  "c":{"raw_outputscale": {"mean":-2.163, "std":2.448}},
                  "noise": {"raw_noise": {"mean":-1.792, "std":3.266}}}
    logables = {}

    total_start = time.time()
    theta_mu = list()
    variances_list = list()
    covar_string = gsr(model.covar_module)
    covar_string_clone = copy.deepcopy(covar_string)
    covar_string_clone = covar_string_clone.replace("(", "")
    covar_string_clone = covar_string_clone.replace(")", "")
    covar_string_clone = covar_string_clone.replace(" ", "")
    covar_string_clone = covar_string_clone.replace("PER", "PER+PER")
    covar_string_list = [s.split("*") for s in covar_string_clone.split("+")]
    covar_string_list.insert(0, ["LIKELIHOOD"])
    covar_string_list = list(chain.from_iterable(covar_string_list))
    both_PER_params = False
    debug_param_name_list = list()
    for (param_name, param), cov_str in zip(model.named_parameters(), covar_string_list):
        debug_param_name_list.append(param_name)
        # First param is (always?) noise and is always with the likelihood
        if "likelihood" in param_name:
            theta_mu.append(prior_dict["noise"]["raw_noise"]["mean"])
            variances_list.append(prior_dict["noise"]["raw_noise"]["std"])
            continue
        else:
            if cov_str == "PER" and not both_PER_params:
                theta_mu.append(prior_dict[cov_str][param_name.split(".")[-1]]["mean"])
                variances_list.append(prior_dict[cov_str][param_name.split(".")[-1]]["std"])
                both_PER_params = True
            elif cov_str == "PER" and both_PER_params:
                theta_mu.append(prior_dict[cov_str][param_name.split(".")[-1]]["mean"])
                variances_list.append(prior_dict[cov_str][param_name.split(".")[-1]]["std"])
                both_PER_params = False
            else:
                try:
                    theta_mu.append(prior_dict[cov_str][param_name.split(".")[-1]]["mean"])
                    variances_list.append(prior_dict[cov_str][param_name.split(".")[-1]]["std"])
                except:
                    import pdb
                    pdb.set_trace()
        prev_cov = cov_str
    theta_mu = torch.tensor(theta_mu)
    theta_mu = theta_mu.unsqueeze(0).t()

    # theta_mu is a vector of parameter priors
    #theta_mu = torch.tensor([1 for p in range(len(params_list))]).reshape(-1,1)

    # sigma is a matrix of variance priors
    sigma = torch.diag(torch.Tensor(variances_list))


    STAN_code = generate_STAN_code(covar_string, debug_param_name_list, covar_string_list)
    # Required data for the STAN model:
    """
    int N;
    int D;
    array[N] real x;
    vector[N] y;
    vector[D] t_mu;
    cov_matrix[D, D] t_sigma;
    """
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
    seed = random.randint(0, 1000000)
    print(STAN_data)
    print(STAN_code)
    posterior = stan.build(STAN_code, data=STAN_data, random_seed=seed)
    end = time.time()
    STAN_model_generation_time = end - start
    if num_draws is None:
       raise("Number of draws not specified")
    start = time.time()

    fit = posterior.sample(num_chains=1, num_samples=num_draws)
    end = time.time()
    STAN_MCMC_sampling_time = end - start

    # Use the sampled parameters to reconstruct the mean and cov. matr.
    # Average(?) to get the actual posterior likelihood
    post_frame = fit.to_frame()
    manual_lp_list = list()
    bad_entries = 0

    start = time.time()
    # Iterate over chain
    for sample in post_frame[list(fit.constrained_param_names)].iterrows():
        # Iterate over kernel parameters
        # Main assumption: Kernel parameters are stored in order of kernel
        # appearance from left to right, just like in STAN
        # Each theta corresponds to exactly one model parameter
        for model_param, sampled_param in zip(model.parameters(), sample[1]):
            model_param.data = torch.full_like(model_param.data, sampled_param)


        try:
            # Compare this to the likelihood of y given mean and covar (+ noise)
            like_mean = torch.zeros(len(model.train_inputs[0]))

            like_cov_matr = torch.eye(len(model.train_inputs[0].tolist())) * likelihood.noise + model.covar_module(model.train_inputs[0])
            like_cov_matr += torch.eye(len(model.train_inputs[0].tolist())) * 1e-4 # Jitter
            like_cov_chol = torch.linalg.cholesky(like_cov_matr.evaluate())
            like_dist = torch.distributions.multivariate_normal.MultivariateNormal(like_mean, scale_tril=like_cov_chol)
            manual_lp_list.append(like_dist.log_prob(model.train_targets))
        except Exception as e:
            bad_entries += 1
            print(e)
            print(list(model.named_parameters()))
            #print(torch.linalg.eig(like_cov_matr.evaluate()))

    end = time.time()
    likelihood_approximation_time = end - start
    total_time = end - total_start

    logables["Kernel code"] = generate_STAN_kernel(covar_string, debug_param_name_list, covar_string_list)
    logables["seed"] = seed
    logables["Likelihood time"] = likelihood_approximation_time
    logables["Model compile time"] = STAN_model_generation_time
    logables["Sampling time"] = STAN_MCMC_sampling_time
    logables["Total time"] = total_time
    logables["Bad entries"] = bad_entries
    logables["likelihood approximation"] = torch.mean(torch.Tensor(manual_lp_list))
    print(f"Num bad entries: {bad_entries}")
    return torch.mean(torch.Tensor(manual_lp_list)), logables


