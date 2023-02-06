import os
import pickle
import torch
import sys
sys.path.append("..")
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

class Experiment:

    def __init__(self, experiment_keyword, repititions : int, attributes :dict = None):
        self.experiment_keyword = experiment_keyword
        self.DEBUG = (self.experiment_keyword == "")
        self.results = {i:{} for i in range(repititions)}
        if not attributes is None:
            self.results["attributes"] = attributes
        self.current_result = 0
        self.loss = None
        return

    def set_current_result(self, res_num):
        self.current_result = res_num
        return

    def print_log(self, text, filename=None):
        if filename is None:
            filename = f"{self.experiment_keyword}"
        print(text)
        #if not self.DEBUG:
        #    with open(filename, "a") as f:
        #        f.write(text+"\n")
        return

    def store_result(self, key, result):
        self.results[self.current_result][key] = result
        return

    def write_results(self, filename=None):
        if filename is None:
            filename = f"{self.experiment_keyword}.pickle"
        with open(filename, 'wb') as fh:
            pickle.dump(self.results, fh)


    def plot_model(self, description:str, NUM_TASKS:int, test_x, train_x, train_y, lower, upper, mean, ylim=[-30, 30], orig_data=None):
        """
        param description: 1. The key which is used to name the model as an svg file.
                           2. Naming of the plots (i.e. title) is based on this parameter
        """
        if NUM_TASKS == 5:
            f, (y1_ax, y2_ax, y3_ax, y4_ax, y5_ax) = plt.subplots(int(1), int(5), figsize=(int(15), int(4)))
        if NUM_TASKS == 3:
            f, (y1_ax, y2_ax, y3_ax) = plt.subplots(int(1), int(3), figsize=(int(15), int(4)))
        elif NUM_TASKS == 2:
            f, (y1_ax, y2_ax) = plt.subplots(int(1), int(2), figsize=(int(15), int(4)))

        # Plot training data as black stars
        y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
        # Predictive mean as blue line
        y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
        if orig_data is not None:
            y1_ax.plot(train_x.numpy(), orig_data[:, 0].numpy(), 'r', alpha=float(0.6))
        # Shade in confidence
        y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
        y1_ax.set_ylim(ylim)
        y1_ax.legend(['Mean', 'Confidence'])
        y1_ax.set_title(description)

        # Plot training data as black stars
        y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
        # Predictive mean as blue line
        y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
        if orig_data is not None:
            y2_ax.plot(train_x.numpy(), orig_data[:, 1].numpy(), 'r', alpha=float(0.6))
        # Shade in confidence
        y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
        y2_ax.set_ylim(ylim)
        y2_ax.legend(['Mean', 'Confidence'])
        y2_ax.set_title(description)

        if NUM_TASKS >= 3:
            # Plot training data as black stars
            y3_ax.plot(train_x.detach().numpy(), train_y[:, 2].detach().numpy(), 'k*')
            # Predictive mean as blue line
            y3_ax.plot(test_x.numpy(), mean[:, 2].numpy(), 'b')
            if orig_data is not None:
                y3_ax.plot(train_x.numpy(), orig_data[:, 2].numpy(), 'r', alpha=float(0.6))
            # Shade in confidence
            y3_ax.fill_between(test_x.numpy(), lower[:, 2].numpy(), upper[:, 2].numpy(), alpha=0.5)
            y3_ax.set_ylim(ylim)
            y3_ax.legend(['Mean', 'Confidence'])
            y3_ax.set_title(description)
        if NUM_TASKS >=5:

            # Plot training data as black stars
            y4_ax.plot(train_x.detach().numpy(), train_y[:, 3].detach().numpy(), 'k*')
            # Predictive mean as blue line
            y4_ax.plot(test_x.numpy(), mean[:, 3].numpy(), 'b')
            if orig_data is not None:
                y4_ax.plot(train_x.numpy(), orig_data[:, 3].numpy(), 'r', alpha=float(0.6))
            # Shade in confidence
            y4_ax.fill_between(test_x.numpy(), lower[:, 3].numpy(), upper[:, 3].numpy(), alpha=0.5)
            y4_ax.set_ylim(ylim)
            y4_ax.legend(['Mean', 'Confidence'])
            y4_ax.set_title(description)

            # Plot training data as black stars
            y5_ax.plot(train_x.detach().numpy(), train_y[:, 4].detach().numpy(), 'k*')
            # Predictive mean as blue line
            y5_ax.plot(test_x.numpy(), mean[:, 4].numpy(), 'b')
            if orig_data is not None:
                y5_ax.plot(train_x.numpy(), orig_data[:, 4].numpy(), 'r', alpha=float(0.6))
            # Shade in confidence
            y5_ax.fill_between(test_x.numpy(), lower[:, 4].numpy(), upper[:, 4].numpy(), alpha=0.5)
            y5_ax.set_ylim(ylim)
            y5_ax.legend(['Mean', 'Confidence'])
            y5_ax.set_title(description)
        return f

    def training_loop(self, model, likelihood, train_x, train_y, optimizer="Adam", lr=0.1, train_iterations=300):
        # Stuff for the LBFGS optimizer
        #optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')  # Includes GaussianLikelihood parameters
        """
        """
        #optimizer.step(closure)
        # this is for running the notebook in our testing framework
        import os
        smoke_test = ('CI' in os.environ)
        training_iter = int(2) if smoke_test else int(train_iterations)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loop = "standard_loop"
        elif optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            loop = "standard_loop"
        elif optimizer == 'LBFGS':
            loop = "LBFGS_loop"
            def closure():
                optimizer.zero_grad()
                # Output from model
                output = model(train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)
                self.loss = loss.item()
                loss.backward()
                return loss
            optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        param_dict = {parameter[0]: [] for parameter in model.named_parameters()}
        param_dict['loss'] = []

        if loop == "standard_loop":
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)
                #self.print_log(f"{i+1}/{training_iter}; loss: {loss.item()}")
                param_dict['loss'].append(loss.item())
                if i == training_iter-1:
                    print(f"{i}/{training_iter} \t {loss.item()}")
                #pdb.set_trace()
                loss.backward()
                for parameter in model.named_parameters():
                    param_dict[parameter[0]].append(parameter[1])
                #    if 'covar' in parameter[0]:
                #        param_dict[parameter[0]].append(parameter[1].item())
                #param_dict['noise'].append(likelihood.noise.item())
                #for l in range(len(likelihood.task_noises)):
                #    param_dict['task_noises'][l].append(likelihood.task_noises[l].item())
                optimizer.step()
            return param_dict
        elif loop == "LBFGS_loop":
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                # Calc loss and backprop gradients
                #pdb.set_trace()
                param_dict['loss'].append(self.loss)
                optimizer.step(closure)
                for parameter in model.named_parameters():
                    param_dict[parameter[0]].append(parameter[1])
            return param_dict


