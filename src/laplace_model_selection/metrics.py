import copy
import dynesty
import gpytorch
from laplace_model_selection.gpr.helpFunctions import get_string_representation_of_kernel as gsr
from helpers.util_functions import reparameterize_model, fixed_reinit, prior_distribution, log_normalized_prior, extract_model_parameters
from helpers.training_functions import kernel_parameter_priors, parameter_priors
import itertools
import numpy as np
import pickle
import random
import scipy
import sys
import time
import torch
import os


# https://www.sfu.ca/sasdoc/sashtml/iml/chap11/sect8.htm
# Also https://en.wikipedia.org/wiki/Finite_difference_coefficient
def finite_difference_second_derivative_GP_neg_unscaled_map(model, likelihood, train_x, train_y, h_i_step=5e-2, h_j_step=5e-2, h_i_vec=[0.0, 0.0, 0.0], h_j_vec=[0.0, 0.0, 0.0], prior=None):
    curr_params = torch.tensor(list(model.parameters()))
    mll_fkt = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    if prior is not None:
        theta_mu = prior.mean
        variance = torch.diag(prior.covariance_matrix)
    else:
        theta_mu = None
        variance = None

    while h_i_step > 1e-10 and h_j_step > 1e-10:
        h_i = h_i_step * torch.tensor(h_i_vec)
        h_j = h_j_step * torch.tensor(h_j_vec)
        map_call = lambda : (-mll_fkt(model(train_x), train_y) - log_normalized_prior(model, param_specs=parameter_priors , kernel_param_specs=kernel_parameter_priors, theta_mu=theta_mu, variance=variance))*len(*model.train_inputs)
        try:
            fixed_reinit(model, curr_params+h_i + h_j)
            f_plus = map_call()

            fixed_reinit(model, curr_params+h_i - h_j)
            f1 = map_call()
            fixed_reinit(model, curr_params-h_i + h_j)
            f2 = map_call()

            fixed_reinit(model, curr_params - h_i - h_j)
            f_minus = map_call()

            # Reverse model reparameterization
            fixed_reinit(model, curr_params)

            return (f_plus - f1 - f2 + f_minus) / (4*h_i_step*h_j_step)

        except Exception as E:
            print(f"Precision {h_i_step+h_j_step} too low. Halving precision")
            h_i_step /= 2
            h_j_step /= 2
            pass
    raise ValueError("Finite difference Hessian calculation failed.")


def finite_difference_hessian(model, likelihood, num_params, train_x, train_y,  h_i_step=5e-2, h_j_step=5e-2, prior=None):
    hessian_finite_differences_neg_unscaled_map = np.zeros((num_params, num_params))
    for i, j in itertools.product(range(num_params), range(num_params)):
        halving_factor = 1.0
        h_i_vec = np.zeros(num_params)
        h_j_vec = np.zeros(num_params)
        h_i_vec[i] = 1.0
        h_j_vec[j] = 1.0
        hessian_finite_differences_neg_unscaled_map[i][j] = finite_difference_second_derivative_GP_neg_unscaled_map(model, likelihood, train_x, train_y,  h_i_step=h_i_step, h_j_step=h_j_step, h_i_vec=h_i_vec, h_j_vec=h_j_vec, prior=prior)
        while i == j and hessian_finite_differences_neg_unscaled_map[i][j] < 0 and h_i_step > 1e-10 and h_j_step > 1e-10:
            halving_factor *= 2
            print("Negative diagonal entry in Hessian. Running with smaller step")
            print(f"New precision: {(h_i_step+h_j_step)/halving_factor}")
            h_i_step_temp = h_i_step/halving_factor
            h_j_step_temp = h_j_step/halving_factor
            hessian_finite_differences_neg_unscaled_map[i][j] = finite_difference_second_derivative_GP_neg_unscaled_map(model, likelihood, train_x, train_y,  h_i_step=h_i_step_temp, h_j_step=h_j_step_temp, h_i_vec=h_i_vec, h_j_vec=h_j_vec, prior=prior)
    return hessian_finite_differences_neg_unscaled_map



def reparameterize_and_pos_mll(model, likelihood, theta, train_x, train_y):
    reparameterize_model(model, theta)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    with torch.no_grad():
        return mll(model(train_x), train_y)

class Lap():
    def __init__(self, threshold, prior=None):
        self.threshold = threshold
        self.prior = prior

    def __call__(self, pos_unscaled_optimum, model_parameters, **kwargs):
        logging = kwargs.get("logging", False)
        neg_unscaled_optimum = -pos_unscaled_optimum
        if logging:
            hessian, hessian_logs = self.calc_hessian(neg_unscaled_optimum, model_parameters, **kwargs)
            constructed_eigvals_log, eigval_correction_logs = self.eigenvalue_correction(hessian, self.threshold, **kwargs)
        else:
            hessian = self.calc_hessian(neg_unscaled_optimum, model_parameters, **kwargs)
            constructed_eigvals_log = self.eigenvalue_correction(hessian, self.threshold, **kwargs)
        punish_term = 0.5*(len(model_parameters)*torch.log(torch.tensor(2*torch.pi)) - torch.sum(torch.log(torch.tensor(constructed_eigvals_log))))
        laplace = -neg_unscaled_optimum + punish_term
        if not torch.isfinite(laplace) and laplace > 0:
            import pdb
            pdb.set_trace()
        if logging:
            total_logs = {}
            total_logs.update(hessian_logs)
            total_logs.update(eigval_correction_logs)
            total_logs.update({
                "neg MAP": neg_unscaled_optimum,
                "punish term": punish_term,
                "laplace without replacement": -neg_unscaled_optimum + 0.5*(len(model_parameters)*torch.log(torch.tensor(2*torch.pi)) - torch.logdet(hessian)),
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
        model = kwargs.get("model", None)
        if bool_use_finite_difference_hessian:
            if model is None:
                raise ValueError("Model must be provided when using finite difference Hessian.")
            if self.prior is None:
                raise ValueError("Prior must be provided when using finite difference Hessian.")
        logging = kwargs.get("logging", False)

        if (logging and not model) or bool_use_finite_difference_hessian and not model:
            raise ValueError("Model must be provided when logging or finite differences are active.")

        # Calculate the Hessian using autograd
        if not bool_use_finite_difference_hessian or logging:
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

            hessian_neg_unscaled_map_raw = torch.tensor(hessian_neg_unscaled_map_raw)
            hessian_neg_unscaled_map_raw = hessian_neg_unscaled_map_raw.to(torch.float64)
            hessian_neg_unscaled_map_raw = hessian_neg_unscaled_map_raw + hessian_neg_unscaled_map_raw.t()
            hessian_neg_unscaled_map_raw = hessian_neg_unscaled_map_raw / 2.0
        # Calculate the Hessian using finite differences
        if (bool_use_finite_difference_hessian or logging) and model is not None:
            hessian_neg_unscaled_finite_differences = torch.tensor(finite_difference_hessian(model, model.likelihood, len(model_parameters), model.train_inputs[0], model.train_targets, prior=self.prior))

            hessian_neg_unscaled_finite_differences = hessian_neg_unscaled_finite_differences.to(torch.float64)
            hessian_neg_unscaled_finite_differences = hessian_neg_unscaled_finite_differences + hessian_neg_unscaled_finite_differences.t()
            hessian_neg_unscaled_finite_differences = hessian_neg_unscaled_finite_differences / 2.0

        hessian_to_use = hessian_neg_unscaled_finite_differences if bool_use_finite_difference_hessian else hessian_neg_unscaled_map_raw

        if logging:
            return hessian_to_use, {
                    "Jacobian autograd": jacobian_neg_unscaled_map if (not bool_use_finite_difference_hessian or logging) else None,
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
            [torch.max(val, (torch.exp(torch.tensor(-2*param_punish_term))*(2*torch.pi))) for val in vals], dtype=vals.dtype))
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
    
    def __str__(self):
        return f"Lap_{self.threshold}"

class Lap0(Lap):
    def __init__(self, prior=None):
        super().__init__(threshold=0.0, prior=prior)

    def __str__(self):
        return "Lap0"


class LapAIC(Lap):
    def __init__(self, prior=None):
        super().__init__(threshold=-1.0, prior=prior)

    def __str__(self):
        return "LapAIC"


class LapBIC(Lap):
    def __init__(self, num_data, prior=None):
        self.num_data = num_data
        # threshold is -0.5*log(n) where n is the number of data points
        super().__init__(threshold=-0.5*np.log(num_data), prior=prior)

    def __str__(self):
        return "LapBIC"


class NestedSampling():
    def __init__(self, model, **kwargs):
        self.model = model 

        self.dynamic_sampling = kwargs.get("dynamic_sampling", True)
        self.prior = kwargs.get("prior", None)
        # The number of log likelihood calls
        self.maxcall = kwargs.get("maxcall", sys.maxsize)
        # The maximum number of iterations
        self.maxiter = kwargs.get("maxiter", sys.maxsize)
        self.checkpoint_every = kwargs.get("checkpoint_every", sys.maxsize)
        self.random_seed = kwargs.get("random_seed", random.randint(0, 1000000))

        self.store_samples = kwargs.get("store_samples", False)
        self.store_likelihoods = kwargs.get("store_likelihoods", False)
        self.store_full = kwargs.get("store_full", False)
        self.pickle_directory = kwargs.get("pickle_directory", "")
        self.checkpoint_file = kwargs.get("checkpoint_file", None)
        self.res_file_name = kwargs.get("res_file_name", None)
        self.print_progress = kwargs.get("print_progress", False)


        self.ndim = len(list(model.parameters()))


    def loglike(self, theta_i):
        try:
            log_like = (reparameterize_and_pos_mll(self.model, self.model.likelihood, theta_i, 
                                            self.model.train_inputs[0], 
                                            self.model.train_targets)*len(*self.model.train_inputs)).detach().numpy()
        except Exception as E:
            #print(E)
            log_like = -np.inf
        return log_like

    # Define our prior via the prior transform.
    def prior_transform(self, u):
        """Transforms the uniform random variables `u ~ Unif[0., 1.)`
        to the parameters of interest."""

        x = np.array(u)  # copy u

        prior_theta_mean = self.prior.mean
        prior_theta_cov = self.prior.covariance_matrix

        # Bivariate Normal
        t = scipy.stats.norm.ppf(u)  # convert to standard normal
        Csqrt = np.linalg.cholesky(prior_theta_cov.numpy())
        x = np.dot(Csqrt, t)  # correlate with appropriate covariance
        mu = prior_theta_mean.flatten().numpy()  # mean
        x += mu  # add mean
        return x

    def __call__(self, **kwargs):
        self.logging = kwargs.get("logging", False)
        rng_generator = np.random.default_rng(seed=self.random_seed)
        print(f"Random seed: {self.random_seed}")
        if self.dynamic_sampling:
            # Trying out dynamic sampler
            dsampler = dynesty.DynamicNestedSampler(self.loglike, self.prior_transform, self.ndim, 
                                                    rstate=rng_generator)
            start_time = time.time()
            #dsampler.run_nested(dlogz_init=0.01, maxcall=100000, print_progress=print_progress)# nlive_init=500, nlive_batch=100,
            dsampler.run_nested(dlogz_init=0.01,# nlive_init=500, nlive_batch=100,
                                print_progress=self.print_progress,
                                maxcall=self.maxcall,
                                maxiter=self.maxiter,)
                                # checkpoint_every=checkpoint_every, # checkpoint_file=checkpoint_file
            end_time = time.time()
            res = dsampler.results

        else:
            # Sample from our distribution.
            sampler = dynesty.NestedSampler(self.loglike,
                                            self.prior_transform,
                                            self.ndim,
                                            bound='multi',
                                            sample='auto',
                                            nlive=500)
            sampler.run_nested(dlogz=0.01, print_progress=self.print_progress)
            res = sampler.results
        if self.logging:
            logables = dict()
            logables["Sample time"] = end_time - start_time
            logables["log Z"] = res["logz"][-1]
            logables["log Z err"] = res["logzerr"][-1]
            logables["prior mean"] = self.prior.mean
            logables["prior cov"] = self.prior.covariance_matrix
            logables["dynamic"] = self.dynamic_sampling
            logables["num sampled"] = res.niter
            logables["parameter statistics"] = {"mu": np.mean(res.samples, axis=0),
                                                "std": np.std(res.samples, axis=0)}
            if self.store_likelihoods and not self.store_full:
                logables["log likelihoods"] = res["logl"]
            if self.store_samples and not self.store_full:
                logables["samples"] = res["samples"]
            if self.store_full:
                pickle_filename = f"res_{time.time()}.pkl" if self.res_file_name is None else self.res_file_name
                if not os.path.exists(os.path.join(self.pickle_directory, "Nested_results")):
                    os.makedirs(os.path.join(self.pickle_directory, "Nested_results"))
                full_pickle_path = os.path.join(self.pickle_directory, "Nested_results", pickle_filename)
                pickle.dump(res, open(full_pickle_path, "wb"))
                logables["res file"] = full_pickle_path
            return res.logz[-1], logables
        else:
            return res.logz[-1]

    def __str__(self):
        return "NestedSampling"


class AIC():
    def __init__(self):
        pass

    def __call__(self, pos_unscaled_mll, num_params, **kwargs):
        logging = kwargs.get("logging", False)
        start = time.time()
        aic = 2*num_params - 2*(pos_unscaled_mll)
        end = time.time()
        if logging:
            logables = {"punish term" : 2*num_params,
                        "Total time": end - start,
                        "loss term": 2*(pos_unscaled_mll)}
            return aic, logables
        else:
            return aic

    def __str__(self):
        return "AIC"


class BIC():
    def __init__(self, num_data):
        self.num_data = torch.tensor(num_data)

    def __call__(self, pos_unscaled_mll, num_params, **kwargs):
        logging = kwargs.get("logging", False)
        start = time.time()
        bic = num_params*torch.log(self.num_data) - 2*(pos_unscaled_mll)
        end = time.time()
        if logging:
            logables = {"punish term" : num_params*torch.log(self.num_data),
                        "Total time": end - start,
                        "loss term": 2*(pos_unscaled_mll)}
            return bic, logables
        else:
            return bic

    def __str__(self):
        return "BIC"


#def manual_mll(likelihood):
#    A = lambda pred, train_y:  - 0.5*train_y.T @ pred.covariance_matrix.inverse() @ train_y 
#    B = lambda pred, train_y:  - 0.5*torch.logdet(pred.covariance_matrix)
#    C = lambda pred, train_y:  - 0.5*train_y.size(0)*np.log(2*np.pi)
#
#    mll = lambda latent_pred, train_y: A(likelihood(latent_pred), train_y) + B(likelihood(latent_pred), train_y) + C(likelihood(latent_pred), train_y)
#    return A, B, C, mll
#
def safe_covariance_matrix(cov_matrix, jitter=1e-4, max_tries=5):
    for i in range(max_tries):
        try:
            # Try Cholesky to ensure positive definiteness
            _ = torch.linalg.cholesky(cov_matrix)
            return cov_matrix
        except RuntimeError:
            cov_matrix = cov_matrix + jitter * torch.eye(cov_matrix.size(-1), device=cov_matrix.device)
            jitter *= 10  # Increase jitter if needed
    raise RuntimeError("Covariance matrix not positive definite even after adding jitter.")


def manual_mll(likelihood):
    def A(pred, train_y):
        cov = safe_covariance_matrix(pred.covariance_matrix)
        # instead of torch.linalg.inv(cov) we use torch.linalg.solve for better numerical stability
        return -0.5 * train_y.T @ torch.linalg.solve(cov, train_y)
        #return -0.5 * train_y.T @ torch.linalg.inv(cov) @ train_y

    def B(pred, train_y):
        cov = safe_covariance_matrix(pred.covariance_matrix)
        return -0.5 * torch.logdet(cov)

    def C(pred, train_y):
        return -0.5 * train_y.size(0) * np.log(2 * np.pi)

    mll = lambda latent_pred, train_y: A(likelihood(latent_pred), train_y) + \
                                       B(likelihood(latent_pred), train_y) + \
                                       C(likelihood(latent_pred), train_y)
    return A, B, C, mll




class MAP():
    def __init__(self, logarithmic : bool, scaling : bool):
        self.logarithmic = logarithmic
        self.scaling = scaling

    def __call__(self, model, likelihood, train_x, train_y, prior, given_mll=None, **kwargs):
        logging = kwargs.get("logging", False)
        if given_mll is None:
            A, B, C, mll = manual_mll(likelihood)
        else:
            mll = given_mll
        gradient_needed = kwargs.get("gradient_needed", False)
        with torch.set_grad_enabled(gradient_needed):
            mll_value = mll(model(train_x), train_y) 
            if given_mll:
                mll_value = mll_value* len(*model.train_inputs)
            if not self.logarithmic:
                mll_value = torch.exp(mll_value)
                prior_value = torch.exp(prior.log_prob(extract_model_parameters(model)))
                map = mll_value * prior_value
            else:
                map = mll_value + prior.log_prob(extract_model_parameters(model))
            if self.scaling:
                map = map / len(*model.train_inputs)
            if logging:
                logables = {"scaling": self.scaling,
                            "logarithmic": self.logarithmic,
                            "prior_value": prior.log_prob(extract_model_parameters(model)) if self.logarithmic else torch.exp(prior.log_prob(extract_model_parameters(model))),}
                if given_mll is None:
                    latent_pred = model(train_x)
                    data_fit = A(likelihood(latent_pred), train_y)
                    complexity_term = B(likelihood(latent_pred), train_y) 
                    normalization_constant = C(likelihood(latent_pred), train_y)
                    logables.update({
                        "data fit": data_fit,
                        "complexity term": complexity_term,
                        "normalization constant": normalization_constant,
                    })
                return map, logables 
            else:
                return map

    def __str__(self):
        output_str = "MAP"
        if self.scaling:
            output_str = "scaled " + output_str
        if self.logarithmic:
            output_str = "log "+ output_str
        return output_str


class MLL():
    def __init__(self, logarithmic : bool, scaling : bool):
        self.logarithmic = logarithmic
        self.scaling = scaling

    def __call__(self, model, likelihood, train_x, train_y, given_mll=None, **kwargs):
        logging = kwargs.get("logging", False)
        if given_mll is None:
            A, B, C, mll = manual_mll(likelihood)
        else:
            mll = given_mll
        gradient_needed = kwargs.get("gradient_needed", False)
        with torch.set_grad_enabled(gradient_needed):
            mll_value = mll(model(train_x), train_y) 
            if given_mll:
                mll_value = mll_value* len(*model.train_inputs)
            if not self.logarithmic:
                mll_value = torch.exp(mll_value)
            if self.scaling: 
                mll_value = mll_value / len(*model.train_inputs)
            if logging:
                logables = {"scaling": self.scaling,
                            "logarithmic": self.logarithmic,}
                if given_mll is None:
                    latent_pred = model(train_x)
                    data_fit = A(likelihood(latent_pred), train_y)
                    complexity_term = B(likelihood(latent_pred), train_y) 
                    normalization_constant = C(likelihood(latent_pred), train_y)
                    logables.update({
                        "data fit": data_fit,
                        "complexity term": complexity_term,
                        "normalization constant": normalization_constant,
                    })
                return mll_value, logables 
            else:
                return mll_value

    def __str__(self):
        output_str = "ML"
        if self.scaling:
            output_str = "scaled " + output_str
        if self.logarithmic:
            output_str = "log "+ output_str
        return output_str