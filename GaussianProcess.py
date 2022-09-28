import gpytorch as gpt
import torch
import matplotlib.pyplot as plt
import copy
from globalParams import options, hyperparameter_limits
from helpFunctions import get_kernels_in_kernel_expression






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

    def train_model(self):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([{"params": self.parameters()}], lr=options["training"]["learning_rate"])
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(options["training"]["max_iter"]):
            optimizer.zero_grad()
            output = self.__call__(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()
            if options["training"]["print_training_output"]:
                parameter_string = ""
                for param_name, param in self.covar_module.named_parameters():
                    parameter_string += f"{param_name:20}: raw: {param.item():10}, transformed: {self.covar_module.constraint_for_parameter_name(param_name).transform(param).item():10}\n"
                parameter_string += f"{'noise':20}: raw: {self.likelihood.raw_noise.item():10}, transformed: {self.likelihood.noise.item():10}"
                print(
                f"HYPERPARAMETER TRAINING: Iteration {i} - Loss: {loss.item()}  \n{parameter_string}")
            optimizer.step()

    def get_current_loss(self):
        self.train()
        self.likelihood.train()
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        output = self.__call__(self.train_inputs[0])
        loss = -mll(output, self.train_targets)
        return loss.item()

    def get_ll(self, X = None, Y = None):
        self.eval()
        self.likelihood.eval()
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        output = self.__call__(X)
        return torch.exp(mll(output, Y)).item()

    def optimize_hyperparameters(self):
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
            self.train_model()
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

    def plot_model(self, return_figure = False, figure = None, ax = None):
        self.eval()
        self.likelihood.eval()

        interval_length = torch.max(self.train_inputs[0]) - torch.min(self.train_inputs[0])
        shift = interval_length * options["plotting"]["border_ratio"]
        test_x = torch.linspace(torch.min(self.train_inputs[0]) - shift, torch.max(self.train_inputs[0]) + shift, options["plotting"]["sample_points"], dtype=torch.float64)

        with torch.no_grad(), gpt.settings.fast_pred_var():
            observed_pred = self.likelihood(self(test_x))

        with torch.no_grad():
            if not (figure and ax):
                figure, ax = plt.subplots(1, 1, figsize=(8, 6))


            lower, upper = observed_pred.confidence_region()
            ax.plot(self.train_inputs[0].numpy(), self.train_targets.numpy(), 'k.', zorder=2)
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

