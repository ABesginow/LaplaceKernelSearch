import gpytorch as gpt
import torch
import numpy as np
from GaussianProcess import ExactGPModel
from helpFunctions import get_string_representation_of_kernel as gsr, clean_kernel_expression, print_formatted_hyperparameters
from helpFunctions import amount_of_base_kernels
from gpytorch.kernels import ScaleKernel
import threading
import copy

from globalParams import options

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------HELP FUNCTIONS----------------------------------------------
# ----------------------------------------------------------------------------------------------------
def replace_internal_kernels(base, kernels):
    ret = []
    if not hasattr(base, "kernels"): # or hasattr(base, "base_kernel")):
        return [k for k in kernels if k is not None and not gsr(k)==gsr(base)]
    elif hasattr(base, "kernels"): # operators
        for position in range(len(base.kernels)):
            for k in replace_internal_kernels(base.kernels[position], kernels):
                new_expression = copy.deepcopy(base)
                new_expression.kernels[position] = k
                ret.append(new_expression)
            if None in kernels:
                if len(base.kernels) > 2:
                    new_expression = copy.deepcopy(base)
                    new_expression.kernels.pop(position)
                    ret.append(new_expression)
                else:
                    ret.append(base.kernels[position])
#    elif hasattr(base, "base_kernel"): # scaleKernel
#        for k in replace_internal_kernels(base.base_kernel, kernels):
#            new_expression = copy.deepcopy(base)
#            new_expression.base_kernel = k
#            ret.append(new_expression)
    for expression in ret:
        clean_kernel_expression(expression)
    return ret

def extend_internal_kernels(base, kernels, operations):
    ret = []
    for op in operations:
        ret.extend([op(base, k) for k in kernels])
    if hasattr(base, "kernels"):
        for position in range(len(base.kernels)):
            for k in extend_internal_kernels(base.kernels[position], kernels, operations):
                new_expression = copy.deepcopy(base)
                new_expression.kernels[position] = k
                ret.append(new_expression)
#    elif hasattr(base, "base_kernel"):
#        for k in extend_internal_kernels(base.base_kernel, kernels, operations):
#            new_expression = copy.deepcopy(base)
#            new_expression.base_kernel = k
#            ret.append(new_expression)
    return ret


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------CANDIDATE FUNCTIONS-----------------------------------------
# ----------------------------------------------------------------------------------------------------
def create_candidates_CKS(base, kernels, operations):
    ret = []
    ret.extend(list(set(extend_internal_kernels(base, kernels, operations))))
    ret.extend(list(set(replace_internal_kernels(base, kernels))))
    return ret

def create_candidates_AKS(base, kernels, operations, max_complexity=5):
    ret = []
    if max_complexity and amount_of_base_kernels(base) < max_complexity:
        ret.extend(extend_internal_kernels(base, kernels, operations))
    ret.extend((replace_internal_kernels(base, [None] + kernels)))
    ret.append(base)
    ret = list(set(ret))
    return ret
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------EVALUATE FUNCTIONS------------------------------------------
# ----------------------------------------------------------------------------------------------------
def evaluate_performance_via_likelihood(model):
    return - model.get_current_loss()


def calculate_laplace(model, loss_of_model, variances_list=None):
    num_of_observations = len(*model.train_inputs)
    # Save a list of model parameters and compute the Hessian of the MLL
    params_list = [p for p in model.parameters()]
    mll         = (num_of_observations * (-loss_of_model))
    env_grads   = torch.autograd.grad(mll, params_list, retain_graph=True, create_graph=True)
    hess_params = []
    for i in range(len(env_grads)):
            hess_params.append(torch.autograd.grad(env_grads[i], params_list, retain_graph=True))

    # theta_mu is a vector of parameter priors
    """
    SE length    1.607
    PER length   1.473
    PER period   0.920
    LIN variance 0.374
    C scale      0.427
    Noise noise  0.531
    """
    theta_mu = torch.tensor([1 for p in range(len(params_list))]).reshape(-1,1)

    # sigma is a matrix of variance priors
    """
    SE length    1.650
    PER length   1.582
    PER period   0.690
    LIN variance 0.309
    C scale      0.754
    Noise noise  0.384
    """
    sigma = []
    if variances_list is None:
        variances_list = [4 for i in range(len(list(model.parameters())))]
    for i in range(len(params_list)):
        line = (np.zeros(len(params_list))).tolist()
        line[i] = variances_list[i]
        sigma.append(line)
    sigma = torch.tensor(sigma)


    params = torch.tensor(params_list).clone().reshape(-1,1)
    hessian = torch.tensor(hess_params).clone()


    # Here comes what's wrapped in the exp-function:
    thetas_added = params+theta_mu
    thetas_added_transposed = (params+theta_mu).reshape(1,-1)
    middle_term = (sigma.inverse()-hessian).inverse()
    matmuls    = torch.matmul( torch.matmul( torch.matmul( torch.matmul(thetas_added_transposed, sigma.inverse()), middle_term ), hessian ), thetas_added )


    # We can calculate by taking the log of the fraction:
    #fraction = 1 / (sigma.inverse()-hessian).det().sqrt() / sigma.det().sqrt()
    #laplace = mll + torch.log(fraction) + (-1/2) * matmuls

    # Conveniently we can also just express the fraction as a sum:
    laplace = mll - (1/2)*torch.log(sigma.det()) - (1/2)*torch.log( (sigma.inverse()-hessian).det() )  + (-1/2) * matmuls

    return laplace


def calculate_mc(model, likelihood, number_of_draws=1000, mean=0, std_deviation=2, print_steps=False, scale_data = False):

    observations_x = model.train_inputs
    num_of_observations = len(*model.train_inputs)
    if type(observations_x) == tuple:
        observations_x = observations_x[0]
    observations_y = model.train_targets
    # We copy the model to keep the original model unchanged while we assign different parameter values to the copy
    model_mc        = copy.deepcopy(model)
    likelihood_mc   = copy.deepcopy(likelihood)

    # How many parameters does the model have?
    params_list     = [p for p in model_mc.parameters()]
    num_of_params   = len(params_list)

    #TODO We likely have to rewrite this code to deal with same parameters multiple times

    # We create an array of tensors
    # Each tensor has random values to use for one model parameter
    # These values are normally distributed according to suitable means and standard deviations and not sorted
    random_values   = [None] * num_of_params
    for i in range(num_of_params):
        if (i==0):        # raw noise
            random_values[i] = torch.tensor(  np.random.normal(0.0, 4, number_of_draws)  )
        elif (i==1):      # mean constant
            random_values[i] = torch.tensor(  np.random.normal(0.0, 2, number_of_draws)  )
        elif (i==2):      # raw lengthscale   (or possibly raw offset)
            random_values[i] = torch.tensor(  np.random.normal(0.0, 2, number_of_draws)  )
            #random_values[i] = torch.tensor(  np.random.normal(-2.5, 1.5, number_of_draws)  )
        elif (i==3):      # raw variance      (or possibly raw lengthscale or raw alpha)
            random_values[i] = torch.tensor(  np.random.normal(-2.5, 1.5, number_of_draws)  )
            #random_values[i] = torch.tensor(  np.random.normal(0.0, 2, number_of_draws)  )
        elif (i==4):      # raw period length (or possibly raw lengthscale)
            random_values[i] = torch.tensor(  np.random.normal(2.5, 1.5, number_of_draws)  )
        elif (i==5):      # raw alpha
            random_values[i] = torch.tensor(  np.random.normal(0.0, 2, number_of_draws)  )
        else:
            random_values[i] = torch.tensor(  np.random.normal(mean, std_deviation, number_of_draws)  )

    # An array to store the log-likelihoods in later
    mll_array = []

    for num_draw in range(number_of_draws):
        num_param = 0
        for param_name, param in model_mc.named_parameters():
            param.data.fill_(random_values[num_param][num_draw])
            num_param += 1

        # We can print the parameter values of every round to validate that we are assigning them correctly:
        if print_steps == True:
            print('\nDraw number:  ', num_draw+1)
            print('Randomly assigned parameters:')
            for param_name, param in model_mc.named_parameters():
                print(param.item())

        # Switching to train mode (but we are not actually doing any training)
        model_mc.train()
        likelihood_mc.train()

        mll_mc = gpt.mlls.ExactMarginalLogLikelihood(likelihood_mc, model_mc)

        output_mc = model_mc(observations_x)

        #if scale_data == True:
        #    loss_mc   = -mll_mc(output_mc, observations_y_transformed)
        #else:
        loss_mc   = -mll_mc(output_mc, observations_y)

        # We can look at the loss of our MC model:
        if print_steps == True:
            print('loss of the test copy:       ', loss_mc.item())

        # We save the log-likelihoods of our model into an array
        mll_array.append(num_of_observations * (-loss_mc.item()))

    # We restore the non-logarithmic likelihoods
    # Then we take the mean of these
    # And eventually, the natural logarithm of the mean
    # ... Luckily, Numpy can handle likelihoods of even 10^50 without big issues or inaccuracies.
    #return np.log(np.mean( np.exp(mll_array) ))

    max_mll_array = max(mll_array)
    max_mll_array = np.array(max_mll_array)
    mll_shifted = mll_array-max_mll_array

    return np.log(np.mean(np.exp(mll_shifted))) + max_mll_array





# ----------------------------------------------------------------------------------------------------
# ------------------------------------------- CKS ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------
def CKS(X, Y, likelihood, base_kernels, list_of_variances=None,  experiment=None, iterations=3, metric="MLL", **kwargs):
    operations = [gpt.kernels.AdditiveKernel, gpt.kernels.ProductKernel]
    candidates = base_kernels.copy()
    best_performance = dict()
    models = dict()
    performance = dict()
    threads = list()
    model_steps = list()
    performance_steps = list()
    loss_steps = list()
    for i in range(iterations):
        for k in candidates:
            models[gsr(k)] = ExactGPModel(X, Y, copy.deepcopy(likelihood), copy.deepcopy(k))
            if options["kernel search"]["multithreading"]:
                threads.append(threading.Thread(target=models[gsr(k)].optimize_hyperparameters))
                threads[-1].start()
            else:
                try:
                    models[gsr(k)].optimize_hyperparameters()
                except:
                    continue
        for t in threads:
            t.join()
        for k in candidates:
            if metric == "Laplace":
                performance[gsr(k)] = calculate_laplace(models[gsr(k)], models[gsr(k)].get_current_loss())
            if metric == "MC":
                try:
                    performance[gsr(k)] = calculate_mc(models[gsr(k)], models[gsr(k)].likelihood)
                except:
                    performance[gsr(k)] = np.NINF
            if metric == "AIC":
                try:
                    log_loss = -models[gsr(k)].get_current_loss() * models[gsr(k)].train_inputs[0].numel()
                    performance[gsr(k)] = 2*log_loss + 2*sum(p.numel() for p in models[gsr(k)].parameters() if p.requires_grad)
                except:
                    performance[gsr(k)] = np.NINF
            elif metric == "MLL":
                try:
                    performance[gsr(k)] = evaluate_performance_via_likelihood(models[gsr(k)]).detach().numpy()
                except:
                    performance[gsr(k)] = np.NINF
            # Add variances list as parameter somehow
            if options["kernel search"]["print"]:
                print(f"KERNEL SEARCH: iteration {i} checking {gsr(k)}, loss {-performance[gsr(k)]}")
        if len(best_performance) > 0:
            if best_performance["performance"] >= max(performance.values()):
                if options["kernel search"]["print"]:
                    print("KERNEL SEARCH: no gain through additional kernel length, stopping search")
                break
        best_model = models[max(performance, key=performance.__getitem__)]
        best_performance = {"model": (gsr(best_model.covar_module), best_model.state_dict()), "performance": max(performance.values())}
        model_steps.append(gsr(best_model))
        performance_steps.append(best_performance)
        loss_steps.append(best_model.get_current_loss())
        candidates = create_candidates_CKS(best_model.covar_module, base_kernels, operations)
    if options["kernel search"]["print"]:
        print(f"KERNEL SEARCH: kernel search concluded, optimal expression: {gsr(best_model.covar_module)}")
    return best_model, best_model.likelihood, model_steps, performance_steps, loss_steps

