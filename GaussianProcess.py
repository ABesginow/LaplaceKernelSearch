import gpytorch as gpt
import torch
import matplotlib.pyplot as plt
import copy
from globalParams import options, hyperparameter_limits
from helpFunctions import get_kernels_in_kernel_expression





def log_prior(model, theta_mu=None, sigma=None):
    # params - 
    # TODO de-spaghettize this once the priors are coded properly
    #prior_dict = {"SE": {"raw_lengthscale": {"mean": 0.891, "std": 2.195}},
    #              "MAT": {"raw_lengthscale": {"mean": 1.631, "std": 2.554}},
    #              "PER": {"raw_lengthscale": {"mean": 0.338, "std": 2.636}, 
    #                      "raw_period_length": {"mean": 0.284, "std": 0.902}},
    #              "LIN": {"raw_variance": {"mean": -1.463, "std": 1.633}},
    #              "c": {"raw_outputscale": {"mean": -2.163, "std": 2.448}},
    #              "noise": {"raw_noise": {"mean": -1.792, "std": 3.266}}}

    prior_dict = {  'SE': {'raw_lengthscale': {"mean": -0.21221139138922668, "std":1.8895426067756804}},
                    'MAT52': {'raw_lengthscale':{"mean": 0.7993038925994188, "std":2.145122566357853 } },
                    'MAT32': {'raw_lengthscale':{"mean": 1.5711054238673443, "std":2.4453761235991216 } },
                    'RQ': {'raw_lengthscale':{"mean": -0.049841950913676276, "std":1.9426354614713097 }, 
                            'raw_alpha':{"mean": 1.882148553921053, "std":3.096431944989054 } },
                    'PER':{'raw_lengthscale':{"mean": 0.7778461197268618, "std":2.288946656544974 },
                            'raw_period_length':{"mean": 0.6485334993738499, "std":0.9930632050553377 } },
                    'LIN':{'raw_variance':{"mean": -0.8017903983055685, "std":0.9966569921354465 } },
                    'c':{'raw_outputscale':{"mean": -1.6253091096349706, "std":2.2570021716661923 } },
                    'noise': {'raw_noise':{"mean": -3.51640656386717, "std":3.5831320474767407 }}}

    variances_list = list()
    debug_param_name_list = list()
    theta_mu = list()
    params = list()
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
        params.append(param.item())
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
    sigma = torch.diag(torch.Tensor(variances_list))

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
        optimizer = torch.optim.Adam([{"params": self.parameters()}], lr=options["training"]["learning_rate"])
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        if with_Adam:
            for i in range(options["training"]["max_iter"]):
                optimizer.zero_grad()
                output = self.__call__(self.train_inputs[0])
                loss = -mll(output, self.train_targets)
                if MAP:
                    log_p = log_prior(self)
                    loss -= log_p 
                loss.backward()
                if options["training"]["print_training_output"]:
                    parameter_string = ""
                    for param_name, param in self.covar_module.named_parameters():
                        parameter_string += f"{param_name:20}: raw: {param.item():10}, transformed: {self.covar_module.constraint_for_parameter_name(param_name).transform(param).item():10}\n"
                    parameter_string += f"{'noise':20}: raw: {self.likelihood.raw_noise.item():10}, transformed: {self.likelihood.noise.item():10}"
                    print(
                    f"HYPERPARAMETER TRAINING: Iteration {i} - Loss: {loss.item()}  \n{parameter_string}")
                optimizer.step()

        if with_BFGS:
            # Additional BFGS optimization to better ensure optimal parameters
            LBFGS_optimizer = torch.optim.LBFGS(self.parameters(), line_search_fn='strong_wolfe')
            # define closure
            def closure():
                LBFGS_optimizer.zero_grad()
                output = self.__call__(self.train_inputs[0])
                loss = -mll(output, self.train_targets)
                if MAP:
                    log_p = log_prior(self)
                    loss -= log_p 
                LBFGS_optimizer.zero_grad()
                loss.backward()
                return loss

            #loss = closure()
            #loss.backward()

            #for i in range(training_iter):
                # perform step and update curvature
                #LBFGS_opts = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            LBFGS_optimizer.step(closure)



    def get_current_loss(self):
        self.train()
        self.likelihood.train()
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        output = self.__call__(self.train_inputs[0])
        loss = -mll(output, self.train_targets)
        return loss

    def get_ll(self, X = None, Y = None):
        self.eval()
        self.likelihood.eval()
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        output = self.__call__(X)
        return torch.exp(mll(output, Y)).item()

    def optimize_hyperparameters(self, with_BFGS=False, with_Adam=True, MAP=False):
        """
        find optimal hyperparameters either by BO or by starting from random initial values multiple times, using an optimizer every time
        and then returning the best result
        """
        # setup
        best_loss = 1e400
        optimal_parameters = dict()
        limits = hyperparameter_limits
        # start runs
        for iteration in range(options["training"]["restarts"]+1):
            # optimize and determine loss
            self.train_model(with_BFGS=with_BFGS, with_Adam=with_Adam, MAP=MAP)
            current_loss = self.get_current_loss()
            # check if the current run is better than previous runs
            if current_loss < best_loss:
                # if it is the best, save all used parameters
                best_loss = current_loss
                for param_name, param in self.named_parameters():
                    optimal_parameters[param_name] = copy.deepcopy(param)

            # set new random inital values
            self.likelihood.noise_covar.noise = torch.rand(1) * (limits["Noise"][1] - limits["Noise"][0]) + limits["Noise"][0]
            #self.mean_module.constant = torch.rand(1) * (limits["Mean"][1] - limits["Mean"][0]) + limits["Mean"][0]
            for kernel in get_kernels_in_kernel_expression(self.covar_module):
                hypers = limits[kernel._get_name()]
                for hyperparameter in hypers:
                    new_value = torch.rand(1) * (hypers[hyperparameter][1] - hypers[hyperparameter][0]) + hypers[hyperparameter][0]
                    setattr(kernel, hyperparameter, new_value)

            # print output if enabled
            if options["training"]["print_optimizing_output"]:
                print(f"HYPERPARAMETER OPTIMIZATION: Random Restart {iteration}: loss: {current_loss}, optimal loss: {best_loss}")

        # finally, set the hyperparameters those in the optimal run
        self.initialize(**optimal_parameters)

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

