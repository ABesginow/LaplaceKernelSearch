import copy
from GaussianProcess import ExactGPModel
from globalParams import options
import gpytorch as gpt
from gpytorch.kernels import ScaleKernel
from helpFunctions import get_string_representation_of_kernel as gsr, clean_kernel_expression, print_formatted_hyperparameters
from helpFunctions import amount_of_base_kernels, get_kernels_in_kernel_expression
from itertools import chain
import numpy as np
import re
import random
from scipy.special import lambertw
import stan
import time
import torch
import threading
from metrics import calculate_AIC, calculate_laplace, calculate_mc_STAN, calculate_BIC


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

# ----------------------------------------------------------------------------------------------------
# ------------------------------------------- CKS ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------
def CKS(X, Y, likelihood, base_kernels, list_of_variances=None,  experiment=None, iterations=3, metric="MLL", BFGS=True, num_draws=None, param_punish_term = None, **kwargs):
    operations = [gpt.kernels.AdditiveKernel, gpt.kernels.ProductKernel]
    candidates = base_kernels.copy()
    best_performance = dict()
    models = dict()
    performance = dict()
    threads = list()
    model_steps = list()
    performance_steps = list()
    loss_steps = list()
    logables = list()
    explosion_counter = 0
    total_counter = 0
    for i in range(iterations):
        for k in candidates:
            models[gsr(k)] = ExactGPModel(X, Y, copy.deepcopy(likelihood), copy.deepcopy(k))
            if not metric == "MC":
                if options["kernel search"]["multithreading"]:
                    threads.append(threading.Thread(target=models[gsr(k)].optimize_hyperparameters))
                    threads[-1].start()
                else:
                    try:
                        total_counter += 1
                        train_start = time.time()
                        if metric in ["Laplace", "MAP"]:
                            models[gsr(k)].optimize_hyperparameters(X=X, Y=Y, MAP=True)
                        else:
                            models[gsr(k)].optimize_hyperparameters(X = X, Y = Y)
                        train_end = time.time()
                    except Exception as E:
                        print(E)
                        explosion_counter += 1
                        continue
        for t in threads:
            t.join()
        for k in candidates:
            if metric == "Laplace":
                try:
                    performance[gsr(k)], logs = calculate_laplace(models[gsr(k)], (-models[gsr(k)].curr_loss)*len(
                        *models[gsr(k)].train_inputs), param_punish_term=param_punish_term)
                    logs["iteration"] = i
                    logs["Train time"] = train_end - train_start
                    logables.append(logs)
                except Exception as E:
                    print(E)
                    performance[gsr(k)] = np.NINF
            if metric == "MC":
                #try:
                performance[gsr(k)], logs = calculate_mc_STAN(models[gsr(k)], models[gsr(k)].likelihood, num_draws=num_draws)
                logables.append(logs)

                #except:
                #    import pdb
                #    pdb.post_mortem()
                #    performance[gsr(k)] = np.NINF
            if metric == "AIC":
                performance[gsr(k)], logs = calculate_AIC(-models[gsr(k)].curr_loss* models[gsr(k)].train_inputs[0].numel(), sum(p.numel() for p in models[gsr(k)].parameters() if p.requires_grad))
                performance[gsr(k)] = -performance[gsr(k)]
                logables.append(logs)
            if metric == "BIC":
                try:
                    bic_loss = -models[gsr(k)].curr_loss* models[gsr(k)].train_inputs[0].numel()
                    bic_params = sum(p.numel() for p in models[gsr(k)].parameters() if p.requires_grad)
                    bic_data_count =  torch.tensor(models[gsr(k)].train_inputs[0].numel())
                    performance[gsr(k)], logs = calculate_BIC(bic_loss, bic_params, bic_data_count)
                    performance[gsr(k)] = -performance[gsr(k)]
                    logables.append(logs)
                except Exception as E:
                    performance[gsr(k)] = np.NINF
            elif metric == "MLL" or metric == "MAP":
                try:
                    # If the model was trained using MAP the log normalized prior is already included in the loss
                    # Otherwise it's just the MLL stored in there 
                    performance[gsr(k)] = models[gsr(k)].curr_loss
                except:
                    performance[gsr(k)] = np.NINF
            # Add variances list as parameter somehow
            if options["kernel search"]["print"]:
                print(f"KERNEL SEARCH: iteration {i} checking {gsr(k)}, loss {performance[gsr(k)]}")
                print("--------\n")
        if len(best_performance) > 0:
            if best_performance["performance"] >= max(performance.values()):
                if options["kernel search"]["print"]:
                    print("KERNEL SEARCH: no gain through additional kernel length, stopping search")
                break
        if metric in ["AIC", "BIC"]:
            best_model = models[min(performance, key=performance.__getitem__)]
            best_performance = {"model": (gsr(best_model.covar_module), best_model.state_dict()), "performance": min(performance.values())}
        else:
            best_model = models[max(performance, key=performance.__getitem__)]
            best_performance = {"model": (gsr(best_model.covar_module), best_model.state_dict()), "performance": max(performance.values())}

        model_steps.append(performance)
        performance_steps.append(best_performance)
        try:
            best_current_loss = best_model.curr_loss
        except Exception as E:
            print(E)
            best_current_loss = np.NINF
        loss_steps.append(best_current_loss)
        candidates = create_candidates_CKS(best_model.covar_module, base_kernels, operations)
    if options["kernel search"]["print"]:
        print(f"KERNEL SEARCH: kernel search concluded, optimal expression: {gsr(best_model.covar_module)}")

    return best_model, best_model.likelihood, model_steps, performance_steps, loss_steps, logables, explosion_counter, total_counter

