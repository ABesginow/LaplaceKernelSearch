import copy
import dynesty
import gpytorch
from gpr.helpFunctions import get_string_representation_of_kernel as gsr
import itertools
import numpy as np
import pickle
import random
import scipy
import sys
import time
import torch
import os


def prior_distribution(model, uninformed=False):
    uninformed_prior_dict = {'SE': {'raw_lengthscale' : {"mean": 0. , "std":10.}},
                  'MAT52': {'raw_lengthscale' :{"mean": 0., "std":10. } },
                  'MAT32': {'raw_lengthscale' :{"mean": 0., "std":10. } },
                  'RQ': {'raw_lengthscale' :{"mean": 0., "std":10. },
                          'raw_alpha' :{"mean": 0., "std":10. } },
                  'PER':{'raw_lengthscale':{"mean": 0., "std":10. },
                          'raw_period_length':{"mean": 0., "std":10. } },
                  'LIN':{'raw_variance' :{"mean": 0., "std":10. } },
                  'AFF':{'raw_variance' :{"mean": 0., "std":10. } },
                  'c':{'raw_outputscale':{"mean": 0., "std":10. } },
                  'noise': {'raw_noise':{"mean": 0., "std":10. }}}

    # TODO de-spaghettize this once the priors are coded properly
    prior_dict = {'SE': {'raw_lengthscale' : {"mean": -0.21221139138922668 , "std":1.8895426067756804}},
                  'MAT52': {'raw_lengthscale' :{"mean": 0.7993038925994188, "std":2.145122566357853 } },
                  'MAT32': {'raw_lengthscale' :{"mean": 1.5711054238673443, "std":2.4453761235991216 } },
                  'RQ': {'raw_lengthscale' :{"mean": -0.049841950913676276, "std":1.9426354614713097 },
                          'raw_alpha' :{"mean": 1.882148553921053, "std":3.096431944989054 } },
                  'PER':{'raw_lengthscale':{"mean": 0.7778461197268618, "std":2.288946656544974 },
                          'raw_period_length':{"mean": 0.6485334993738499, "std":0.9930632050553377 } },
                  'LIN':{'raw_variance' :{"mean": -0.8017903983055685, "std":0.9966569921354465 } },
                  'AFF':{'raw_variance' :{"mean": -0.8017903983055685, "std":0.9966569921354465 } },
                  'c':{'raw_outputscale':{"mean": -1.6253091096349706, "std":2.2570021716661923 } },
                  'noise': {'raw_noise':{"mean": -3.51640656386717, "std":3.5831320474767407 }}}
    #prior_dict = {"SE": {"raw_lengthscale": {"mean": 0.891, "std": 2.195}},
    #              "MAT": {"raw_lengthscale": {"mean": 1.631, "std": 2.554}},
    #              "PER": {"raw_lengthscale": {"mean": 0.338, "std": 2.636},
    #                      "raw_period_length": {"mean": 0.284, "std": 0.902}},
    #              "LIN": {"raw_variance": {"mean": -1.463, "std": 1.633}},
    #              "c": {"raw_outputscale": {"mean": -2.163, "std": 2.448}},
    #              "noise": {"raw_noise": {"mean": -1.792, "std": 3.266}}}

    if uninformed:
        prior_dict = uninformed_prior_dict


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
    covar_string_list = list(itertools.chain.from_iterable(covar_string_list))
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
 

def log_normalized_prior(model, theta_mu=None, sigma=None, uninformed=False):
    theta_mu, sigma = prior_distribution(model, uninformed=uninformed) if theta_mu is None or sigma is None else (theta_mu, sigma)
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


def calculate_BIC(pos_unscaled_mll, num_params, num_data):
    start = time.time()
    BIC = num_params*torch.log(num_data) - 2*pos_unscaled_mll
    end = time.time()
    logables = {"punish term" : num_params*torch.log(num_data),
                "Total time": end - start,
                "loss term": 2*pos_unscaled_mll}
    return BIC, logables


def calculate_AIC(pos_unscaled_mll, num_params):
    start = time.time()
    AIC = 2*num_params - 2*pos_unscaled_mll
    end = time.time()
    logables = {"punish term" : 2*num_params,
                "Total time": end - start,
                "loss term": 2*pos_unscaled_mll}
    return AIC, logables


def fixed_reinit(model, parameters: torch.tensor) -> None:
    for i, (param, value) in enumerate(zip(model.parameters(), parameters)):
        param.data = torch.full_like(param.data, value)


# https://www.sfu.ca/sasdoc/sashtml/iml/chap11/sect8.htm
# Also https://en.wikipedia.org/wiki/Finite_difference_coefficient
def finite_difference_second_derivative_GP_neg_unscaled_map(model, likelihood, train_x, train_y, uninformed=False, h_i_step=5e-2, h_j_step=5e-2, h_i_vec=[0.0, 0.0, 0.0], h_j_vec=[0.0, 0.0, 0.0]):
    curr_params = torch.tensor(list(model.parameters()))
    mll_fkt = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    while h_i_step > 1e-10 and h_j_step > 1e-10:
        h_i = h_i_step * torch.tensor(h_i_vec)
        h_j = h_j_step * torch.tensor(h_j_vec)

        try:
            fixed_reinit(model, curr_params+h_i + h_j)
            f_plus = (-mll_fkt(model(train_x), train_y) - log_normalized_prior(model, uninformed=uninformed))*len(*model.train_inputs)

            fixed_reinit(model, curr_params+h_i - h_j)
            f1 = (-mll_fkt(model(train_x), train_y) - log_normalized_prior(model, uninformed=uninformed))*len(*model.train_inputs)
            fixed_reinit(model, curr_params-h_i + h_j)
            f2 = (-mll_fkt(model(train_x), train_y) - log_normalized_prior(model, uninformed=uninformed))*len(*model.train_inputs)

            fixed_reinit(model, curr_params - h_i - h_j)
            f_minus = (-mll_fkt(model(train_x), train_y) - log_normalized_prior(model, uninformed=uninformed))*len(*model.train_inputs)

            # Reverse model reparameterization
            fixed_reinit(model, curr_params)

            return (f_plus - f1 - f2 + f_minus) / (4*h_i_step*h_j_step)

        except Exception as E:
            print(f"Precision {h_i_step+h_j_step} too low. Halving precision")
            h_i_step /= 2
            h_j_step /= 2
            pass
    raise ValueError("Finite difference Hessian calculation failed.")


def finite_difference_hessian(model, likelihood, num_params, train_x, train_y, uninformed=False, h_i_step=5e-2, h_j_step=5e-2):
    hessian_finite_differences_neg_unscaled_map = np.zeros((num_params, num_params))
    for i, j in itertools.product(range(num_params), range(num_params)):
        halving_factor = 1.0
        h_i_vec = np.zeros(num_params)
        h_j_vec = np.zeros(num_params)
        h_i_vec[i] = 1.0
        h_j_vec[j] = 1.0
        hessian_finite_differences_neg_unscaled_map[i][j] = finite_difference_second_derivative_GP_neg_unscaled_map(model, likelihood, train_x, train_y, uninformed=uninformed, h_i_step=h_i_step, h_j_step=h_j_step, h_i_vec=h_i_vec, h_j_vec=h_j_vec)
        while i == j and hessian_finite_differences_neg_unscaled_map[i][j] < 0 and h_i_step > 1e-10 and h_j_step > 1e-10:
            halving_factor *= 2
            print("Negative diagonal entry in Hessian. Running with smaller step")
            print(f"New precision: {(h_i_step+h_j_step)/halving_factor}")
            h_i_step_temp = h_i_step/halving_factor
            h_j_step_temp = h_j_step/halving_factor
            hessian_finite_differences_neg_unscaled_map[i][j] = finite_difference_second_derivative_GP_neg_unscaled_map(model, likelihood, train_x, train_y, uninformed=uninformed, h_i_step=h_i_step_temp, h_j_step=h_j_step_temp, h_i_vec=h_i_vec, h_j_vec=h_j_vec)
    return hessian_finite_differences_neg_unscaled_map




#def Eigenvalue_correction(neg_mll_hessian, param_punish_term):
#    # Appendix E.2
#    vals, vecs = torch.linalg.eigh(neg_mll_hessian)
#    constructed_eigvals = torch.diag(torch.tensor(
#        [max(val, (torch.exp(torch.tensor(-2*param_punish_term))*(2*torch.pi))) for val in vals], dtype=vals.dtype))
#    num_replaced = torch.count_nonzero(vals - torch.diag(constructed_eigvals))
#    corrected_hessian = vecs@constructed_eigvals@vecs.t()
#    return corrected_hessian, torch.diag(constructed_eigvals), num_replaced, vecs
#        
#
#def calculate_laplace(model, pos_unscaled_map, variances_list=None, param_punish_term = -1.0, **kwargs):
#    #torch.set_default_tensor_type(torch.DoubleTensor)
#    theta_mu = kwargs["theta_mu"] if "theta_mu" in kwargs else None
#    bool_use_finite_difference_hessian = kwargs["use_finite_difference_hessian"] if "use_finite_difference_hessian" in kwargs else False
#    uninformed = kwargs["uninformed"] if "uninformed" in kwargs else False
#    logables = {}
#    total_start = time.time()
#    # Save a list of model parameters and compute the Hessian of the MLL
#    params_list = [p for p in model.parameters()]
#    # This is now the negative MLL
#    neg_unscaled_map = -pos_unscaled_map
#    start = time.time()
#    try:
#        jacobian_neg_unscaled_map = torch.autograd.grad(neg_unscaled_map, params_list, retain_graph=True, create_graph=True, allow_unused=True)
#    except Exception as E:
#        print(E)
#        import pdb
#        pdb.set_trace()
#        print(f"E:{E}")
#    hessian_neg_unscaled_map_raw = []
#    # Calcuate -\nabla\nabla log(f(\theta)) (i.e. Hessian of negative log posterior)
#    for i in range(len(jacobian_neg_unscaled_map)):
#        hessian_neg_unscaled_map_raw.append(torch.autograd.grad(jacobian_neg_unscaled_map[i], params_list, retain_graph=True, allow_unused=True))
#    # Calculate the Hessian using finite differences
#    hessian_neg_unscaled_finite_differences = torch.tensor(finite_difference_hessian(model, model.likelihood, len(params_list), model.train_inputs[0], model.train_targets, uninformed=uninformed) if bool_use_finite_difference_hessian else None)
#    end = time.time()
#    derivative_calc_time = end - start
#    if theta_mu is None:
#        theta_mu = []
#    if variances_list is None:
#        variances_list = []
#    debug_param_name_list = [l[0] for l in list(model.named_parameters())] 
#    
#    if variances_list == [] and theta_mu == []:
#        theta_mu, variance = prior_distribution(model, uninformed=uninformed)
#
#    params = torch.tensor(params_list).clone().reshape(-1, 1)
#
#    end = time.time()
#    prior_generation_time = end - start
#
#    hessian_neg_unscaled_map_symmetrized = torch.tensor(hessian_neg_unscaled_map_raw).clone()
#    hessian_neg_unscaled_map_symmetrized = (hessian_neg_unscaled_map_symmetrized + hessian_neg_unscaled_map_symmetrized.t()) / 2
#    hessian_neg_unscaled_map_symmetrized = hessian_neg_unscaled_map_symmetrized.to(torch.float64)
#
#    if param_punish_term == "BIC":
#        param_punish_term = -0.5*torch.log(torch.tensor(model.train_targets.numel()))
#
#
#    hessian_to_use = hessian_neg_unscaled_finite_differences if bool_use_finite_difference_hessian else hessian_neg_unscaled_map_symmetrized
#
#    # Appendix E.1
#    # it says mll, but it's actually the MAP here
#    start = time.time()
#    hessian_neg_unscaled_map_symmetrized_corrected, constructed_eigvals_log, num_replaced, hessian_neg_unscaled_map_symmetrized_eigvecs = Eigenvalue_correction(hessian_to_use, param_punish_term)
#    end = time.time()
#    hessian_correction_time = end - start
#    # punish term = + u/2 * ln(2pi) - 1/2 * ln(det(H))
#    # u = number of hyperparameters
#    #punish_term = 0.5*len(theta_mu)*torch.tensor(1.8378) - 0.5*torch.sum(torch.log(constructed_eigvals_log))
#    punish_term = 0.5*len(theta_mu)*torch.log(torch.tensor(2*np.pi)) - 0.5*torch.sum(torch.log(constructed_eigvals_log))
#    # 1.8378 = ln(2pi)
#    laplace = pos_unscaled_map + punish_term
#    #punish_without_replacement = 0.5*len(theta_mu)*torch.tensor(1.8378) - 0.5*torch.logdet(oldHessian)
#    laplace_without_replacement = pos_unscaled_map + 0.5*len(theta_mu)*torch.log(torch.tensor(2*np.pi)) - 0.5*torch.logdet(hessian_to_use)
#    end = time.time()
#    approximation_time = end - start
#    if param_punish_term == -1.0:
#        if (len(theta_mu) + punish_term) > 1e-4:
#            print("Something went horribly wrong with the c(i)s")
#            import pdb
#            pdb.set_trace()
#            print(constructed_eigvals_log)
#    else:
#        if (len(theta_mu) + punish_term) > len(theta_mu)+1e-4:
#            print("Something went horribly wrong with the c(i)s")
#            import pdb
#            pdb.set_trace()
#            print(constructed_eigvals_log)
#    total_time = end - total_start
#    # Everything worth logging
#    logables["MAP"] = neg_unscaled_map 
#    logables["punish term"] = punish_term 
#    logables["laplace without replacement"] = laplace_without_replacement
#    logables["correction term"] = param_punish_term
#
#    logables["num replaced"] = num_replaced
#    logables["parameter list"] = debug_param_name_list
#    logables["parameter values"] = params
#    logables["Jacobian autograd"] = jacobian_neg_unscaled_map
#    logables["diag(constructed eigvals)"] = constructed_eigvals_log
#    logables["use finite differences"] = bool_use_finite_difference_hessian
#    logables["Hessian finite difference"] = hessian_neg_unscaled_finite_differences
#    logables["Hessian autograd symmetrized"] = hessian_neg_unscaled_map_symmetrized
#    logables["Hessian pre correction"] = hessian_to_use
#    logables["eigenvectors Hessian pre correction"] = hessian_neg_unscaled_map_symmetrized_eigvecs
#    logables["Hessian post correction"] = hessian_neg_unscaled_map_symmetrized_corrected
#    logables["prior mean"] = theta_mu
#    logables["diag(prior var)"] = torch.diag(variance)
#    logables["model evidence approx"] = laplace
#
#    # Everything time related
#    logables["Derivative time"]         = derivative_calc_time
#    logables["Approximation time"]      = approximation_time
#    logables["Correction time"]         = hessian_correction_time
#    logables["Prior generation time"]   = prior_generation_time
#    logables["Total time"]              = total_time
#
#    if not torch.isfinite(laplace) and laplace > 0:
#        import pdb
#        pdb.set_trace()
#
#    #torch.set_default_tensor_type(torch.FloatTensor)
#    return laplace, logables




def reparameterize_model(model, theta):
    for model_param, sampled_param in zip(model.parameters(), theta):
        model_param.data = torch.full_like(model_param.data, float(sampled_param))

def reparameterize_and_pos_mll(model, likelihood, theta, train_x, train_y):
    reparameterize_model(model, theta)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    with torch.no_grad():
        return mll(model(train_x), train_y)


def NestedSampling(model, **kwargs):
    print_progress = kwargs.get("print_progress", False)
    dynamic_sampling = kwargs.get("dynamic_sampling", True)
    store_samples = kwargs.get("store_samples", False)
    store_likelihoods = kwargs.get("store_likelihoods", False)
    store_full = kwargs.get("store_full", False)
    pickle_directory = kwargs.get("pickle_directory", "")
    checkpoint_file = kwargs.get("checkpoint_file", None)
    maxcall = kwargs.get("maxcall", sys.maxsize)
    maxiter = kwargs.get("maxiter", sys.maxsize)
    checkpoint_every = kwargs.get("checkpoint_every", sys.maxsize)
    res_file_name = kwargs.get("res_file_name", None)
    random_seed = kwargs.get("random_seed", None)
    uninformed = kwargs.get("uninformed", False)
    if random_seed is None:
        random_seed = random.randint(0, 1000000)

    prior_theta_mean, prior_theta_cov = prior_distribution(model, uninformed=uninformed)

    # Define the dimensionality of our problem.
    ndim = len(list(model.parameters()))

    def loglike(theta_i):
        try:
            log_like = (reparameterize_and_pos_mll(model, model.likelihood, theta_i, 
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
                            maxcall=maxcall,
                            maxiter=maxiter,)
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


class Lap():
    def __init__(self, threshold, prior=None):
        self.threshold = threshold
        self.prior = prior

    def __call__(self, neg_unscaled_optimum, model_parameters, **kwargs):
        logging = kwargs.get("logging", False)
        if logging:
            hessian, hessian_logs = self.calc_hessian(neg_unscaled_optimum, model_parameters, **kwargs)
            constructed_eigvals_log, eigval_correction_logs = self.eigenvalue_correction(hessian, self.threshold, **kwargs)
        else:
            hessian = self.calc_hessian(neg_unscaled_optimum, model_parameters, **kwargs)
            constructed_eigvals_log = self.eigenvalue_correction(hessian, self.threshold, **kwargs)
        punish_term = 0.5*(len(model_parameters)*torch.log(2*torch.pi) - torch.sum(torch.log(constructed_eigvals_log)))
        laplace = -neg_unscaled_optimum + punish_term
        if not torch.isfinite(laplace) and laplace > 0:
            import pdb
            pdb.set_trace()
        if logging:
            total_logs = {}
            total_logs.update(hessian_logs)
            total_logs.update(eigval_correction_logs)
            total_logs.update({
                "MAP": neg_unscaled_optimum,
                "punish term": punish_term,
                "laplace without replacement": -neg_unscaled_optimum + 0.5*(len(model_parameters)*torch.log(2*torch.pi) - torch.logdet(hessian)),
                "constructed eigvals log": constructed_eigvals_log,
                "correction term": self.threshold,
                "use finite differences": kwargs.get("use_finite_difference_hessian", False),
                "model evidence approx": laplace,
                
            })
            return laplace, total_logs
        else:
            return laplace


    def calc_hessian(self, neg_unscaled_optimum, model_parameters, **kwargs):
        bool_use_finite_difference_hessian = kwargs.get("use_finite_difference_hessian", False)
        if bool_use_finite_difference_hessian:
            model = kwargs.get("model", None)
            if model is None:
                raise ValueError("Model must be provided when using finite difference Hessian.")
            if self.prior is None:
                raise ValueError("Prior must be provided when using finite difference Hessian.")
        logging = kwargs.get("logging", False)
        # Calculate the Hessian using autograd
        if not bool_use_finite_difference_hessian and not logging:
            try:
                jacobian_neg_unscaled_map = torch.autograd.grad(neg_unscaled_optimum, model_parameters, retain_graph=True, create_graph=True, allow_unused=True)
            except Exception as E:
                print(E)
                import pdb
                pdb.set_trace()
                print(f"E:{E}")
            hessian_neg_unscaled_map_raw = []
            # Calcuate -\nabla\nabla log(f(\theta)) (i.e. Hessian of negative log posterior)
            for i in range(len(jacobian_neg_unscaled_map)):
                hessian_neg_unscaled_map_raw.append(torch.autograd.grad(jacobian_neg_unscaled_map[i], model_parameters, retain_graph=True, allow_unused=True))

            hessian_neg_unscaled_map_raw = hessian_neg_unscaled_map_raw.to(torch.float64)
            hessian_neg_unscaled_map_raw = hessian_neg_unscaled_map_raw + hessian_neg_unscaled_map_raw.t()
            hessian_neg_unscaled_map_raw = hessian_neg_unscaled_map_raw / 2.0
        # Calculate the Hessian using finite differences
        elif (bool_use_finite_difference_hessian or not logging) and model is not None:
            hessian_neg_unscaled_finite_differences = torch.tensor(finite_difference_hessian(model, model.likelihood, len(model_parameters), model.train_inputs[0], model.train_targets, prior=self.prior) if bool_use_finite_difference_hessian else None)

            hessian_neg_unscaled_finite_differences = hessian_neg_unscaled_finite_differences.to(torch.float64)
            hessian_neg_unscaled_finite_differences = hessian_neg_unscaled_finite_differences + hessian_neg_unscaled_finite_differences.t()
            hessian_neg_unscaled_finite_differences = hessian_neg_unscaled_finite_differences / 2.0

        hessian_to_use = hessian_neg_unscaled_finite_differences if bool_use_finite_difference_hessian else hessian_neg_unscaled_map_raw

        if logging:
            return hessian_to_use, {
                    "Jacobian autograd": jacobian_neg_unscaled_map,
                    "Hessian autograd symmetrized": hessian_neg_unscaled_map_raw,
                    "Hessian finite difference symmetrized": hessian_neg_unscaled_finite_differences,
                    "Hessian pre correction": hessian_to_use,
                }
        else:
            return hessian_to_use

    def eigenvalue_correction(self, neg_map_hessian, param_punish_term, **kwargs):
        logging = kwargs.get("logging", False)
        # Appendix 
        vals, vecs = torch.linalg.eigh(neg_map_hessian)
        constructed_eigvals = torch.diag(torch.tensor(
            [max(val, (torch.exp(torch.tensor(-2*param_punish_term))*(2*torch.pi))) for val in vals], dtype=vals.dtype))
        if logging:
            num_replaced = torch.count_nonzero(vals - torch.diag(constructed_eigvals))
            corrected_hessian = vecs@constructed_eigvals@vecs.t()
            return torch.diag(constructed_eigvals),{
                "num replaced": num_replaced,
                "eigenvectors Hessian pre correction": vecs,
                "Hessian post correction": corrected_hessian,
                "constructed eigvals log": constructed_eigvals,
            }
        else:
            return torch.diag(constructed_eigvals)
    

class Lap0(Lap):
    def __init__(self, prior=None):
        super().__init__(threshold=0.0, prior=prior)
        pass


class LapAIC(Lap):
    def __init__(self, prior=None):
        super().__init__(threshold=-1.0, prior=prior)
        pass


class LapBIC(Lap):
    def __init__(self, num_data, prior=None):
        self.num_data = num_data
        # threshold is -0.5*log(n) where n is the number of data points
        super().__init__(threshold=-0.5*np.log(num_data), prior=prior)
        pass


class NestedSampling():
    def __init__(self):
        pass

    def __call__(self):
        pass


class AIC():
    def __init__(self):
        pass

    def __call__(self, neg_unscaled_mll, num_params, **kwargs):
        logging = kwargs.get("logging", False)
        start = time.time()
        aic = 2*num_params - 2*(-neg_unscaled_mll)
        end = time.time()
        if logging:
            logables = {"punish term" : 2*num_params,
                        "Total time": end - start,
                        "loss term": 2*(-neg_unscaled_mll)}
            return aic, logables
        else:
            return aic


class BIC():
    def __init__(self, num_data):
        self.num_data = num_data

    def __call__(self, neg_unscaled_mll, num_params, **kwargs):
        logging = kwargs.get("logging", False)
        start = time.time()
        bic = num_params*torch.log(self.num_data) - 2*(-neg_unscaled_mll)
        end = time.time()
        if logging:
            logables = {"punish term" : num_params*torch.log(self.num_data),
                        "Total time": end - start,
                        "loss term": 2*(-neg_unscaled_mll)}
            return bic, logables
        else:
            return bic





class MAP():
    def __init__(self):
        pass

    def __call__(self):
        pass

class MLL():
    def __init__(self):
        pass

    def __call__(self):
        pass