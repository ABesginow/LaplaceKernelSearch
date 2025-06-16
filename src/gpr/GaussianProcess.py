import copy
from globalParams import options, hyperparameter_limits
import gpytorch as gpt
from helpFunctions import get_kernels_in_kernel_expression, get_string_representation_of_kernel as gsr, get_full_kernels_in_kernel_expression
import matplotlib.pyplot as plt
import numpy as np
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import torch
from itertools import chain



def log_normalized_prior(model, theta_mu=None, sigma=None):
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

    variances_list = list()
    debug_param_name_list = list()
    theta_mu = list()
    params = list()
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
        params.append(param.item())
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
    sigma = torch.diag(torch.Tensor(variances_list))
    sigma = sigma@sigma
    prior = torch.distributions.MultivariateNormal(theta_mu.t(), sigma)

    # for convention reasons I'm diving by the number of datapoints
    return prior.log_prob(torch.Tensor(params)).item() / len(*model.train_inputs)

class ExactGPModel(gpt.models.ExactGP):
    """
    A Gaussian Process class.
    This class saves input and target data, the likelihood function and the kernel.
    It can be used to train the hyperparameters or plot the mean function and confidence band.
    """
    def __init__(self, X, Y, likelihood, kernel):
        super(ExactGPModel, self).__init__(X, Y, likelihood)
        self.mean_module = gpt.means.ZeroMean()
        self.covar_module = kernel
        self.curr_loss = np.inf

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpt.settings.fast_pred_var():
            observed_pred = self.likelihood(self(x))
        return observed_pred.mean

    def train_model(self, with_BFGS=False, with_Adam=True, MAP=False):
        self.train()
        self.likelihood.train()
        optimize_hyperparameters(self, self.likelihood, with_BFGS=with_BFGS, with_Adam=with_Adam, MAP=MAP)


    def get_current_loss(self, **kwargs):
        MAP = kwargs.get("MAP", False)
        self.train()
        self.likelihood.train()
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        output = self.__call__(self.train_inputs[0])
        loss -= mll(output, self.train_targets)
        if MAP:
            # log_normalized_prior is in metrics.py 
            log_p = log_normalized_prior(self)
            loss -= log_p
        return loss

    def get_ll(self, X = None, Y = None):
        self.eval()
        self.likelihood.eval()
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        output = self.__call__(X)
        return torch.exp(mll(output, Y)).item()

    def get_log_posterior(self, X = None, Y = None):
        self.train()
        self.likelihood.train()
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        output = self.__call__(X)
        return mll(output, Y) + log_normalized_prior(self)




    def random_reinit(self, model):
        for i, (param, limit) in enumerate(zip(model.parameters(), [{"Noise": hyperparameter_limits["Noise"]},*[hyperparameter_limits[kernel] for kernel in get_full_kernels_in_kernel_expression(model.covar_module)]])):
            covar_text = gsr(self.covar_module)
            param_name = list(limit.keys())[0]
            new_param_value = torch.randn_like(param) * (limit[param_name][1] - limit[param_name][0]) + limit[param_name][0]
            param.data = new_param_value

    # Define the training loop
    def optimize_hyperparameters(self, **kwargs):
        """
        find optimal hyperparameters either by BO or by starting from random initial values multiple times, using an optimizer every time
        and then returning the best result
        """
        log_param_path = kwargs.get("log_param_path", False)
        log_likelihood = kwargs.get("log_likelihood", False)
        random_restarts = kwargs.get("random_restarts", options["training"]["restarts"]+1)
        line_search = kwargs.get("line_search", False)
        BFGS_iter = kwargs.get("BFGS_iter", 50)
        train_iterations = kwargs.get("train_iterations", 0)
        train_x = kwargs.get("X", self.train_inputs)
        train_y = kwargs.get("Y", self.train_targets)
        with_BFGS = kwargs.get("with_BFGS", False)
        history_size = kwargs.get("history_size", 100)
        MAP = kwargs.get("MAP", False)
        prior = kwargs.get("prior", False)
        granso = kwargs.get("granso", True)
        double_precision = kwargs.get("double_precision", False)

        # Set up the likelihood and model
        #likelihood = gpytorch.likelihoods.GaussianLikelihood()
        #model = GPModel(train_x, train_y, likelihood)

        model = self
        likelihood = self.likelihood

        # Define the negative log likelihood
        mll = gpt.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Set up the PyGRANSO optimizer
        opts = pygransoStruct()
        opts.torch_device = torch.device('cpu')
        nvar = getNvarTorch(model.parameters())
        opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
        opts.opt_tol = float(1e-10)
        opts.double_precision = double_precision
        opts.limited_mem_size = int(100)
        opts.globalAD = True
        opts.quadprog_info_msg = False
        opts.print_level = int(0)
        opts.halt_on_linesearch_bracket = False

        # Define the objective function
        def objective_function(model):
            output = model(train_x)
            loss = -mll(output, train_y)
            if MAP:
                # log_normalized_prior is in metrics.py 
                log_p = log_normalized_prior(model)
                loss -= log_p
            return [loss, None, None]

        best_model_state_dict = model.state_dict()
        best_likelihood_state_dict = likelihood.state_dict()

        random_restarts = int(5)
        best_f = np.inf
        for restart in range(random_restarts):
            #print(f"pre training parameters: {list(model.named_parameters())}")
            # Train the model using PyGRANSO
            try:
                soln = pygranso(var_spec=model, combined_fn=objective_function, user_opts=opts)
            except:
                self.random_reinit(model)
                continue
            if soln.final.f < best_f:
                best_f = soln.final.f
                best_model_state_dict = model.state_dict()
                best_likelihood_state_dict = likelihood.state_dict()
            #print(f"post training (final): {list(model.named_parameters())} w. loss: {soln.final.f} (smaller=better)")
            self.random_reinit(model)

        model.load_state_dict(best_model_state_dict)
        likelihood.load_state_dict(best_likelihood_state_dict)

        loss = -mll(model(train_x), train_y)
        if MAP:
            log_p = log_normalized_prior(model)
            loss -= log_p

        #print(f"post training (best): {list(model.named_parameters())} w. loss: {soln.best.f}")
        #print(f"post training (final): {list(model.named_parameters())} w. loss: {soln.final.f}")
        
        #print(torch.autograd.grad(loss, [p for p in model.parameters()], retain_graph=True, create_graph=True, allow_unused=True))
        # Return the trained model
        self.curr_loss = loss

    def eval_model(self):
        pass

    def plot_model(self, return_figure = False, figure = None, ax = None, posterior=False, test_y=None):
        self.eval()
        self.likelihood.eval()

        interval_length = torch.max(self.train_inputs[0]) - torch.min(self.train_inputs[0])
        shift = interval_length * options["plotting"]["border_ratio"]
        test_x = torch.linspace(torch.min(self.train_inputs[0]) - shift, torch.max(self.train_inputs[0]) + shift, options["plotting"]["sample_points"])

        with torch.no_grad(), gpt.settings.fast_pred_var():
            observed_pred = self.likelihood(self(test_x))

        with torch.no_grad():
            if not (figure and ax):
                figure, ax = plt.subplots(1, 1, figsize=(8, 6))


            lower, upper = observed_pred.confidence_region()
            #if not posterior:
            #    ax.plot(self.train_inputs[0].numpy(), self.train_targets.numpy(), 'k.', zorder=2)
            if posterior:
                ax.plot(self.train_inputs[0].numpy(), test_y.numpy(), 'kx', zorder=2)
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), color="b", zorder=3)
            amount_of_gradient_steps = 30
            alpha_min=0.05
            alpha_max=0.8
            alpha=(alpha_max-alpha_min)/amount_of_gradient_steps
            c = ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=alpha+alpha_min, zorder=1).get_facecolor()
            for i in range(1,amount_of_gradient_steps):
                ax.fill_between(test_x.numpy(), (lower+(i/amount_of_gradient_steps)*(upper-lower)).numpy(), (upper-(i/amount_of_gradient_steps)*(upper-lower)).numpy(), alpha=alpha, color=c, zorder=1)
            if options["plotting"]["legend"]:
                ax.plot([], [], 'k.', label="Data")
                ax.plot([], [], 'b', label="Mean")
                ax.plot([], [], color=c, alpha=alpha_max, label="Confidence")
                ax.legend(loc="upper left")
            ax.set_xlabel("Normalized Input")
            ax.set_ylabel("Normalized Output")
        if not return_figure:
            plt.show()
        else:
            return figure, ax

