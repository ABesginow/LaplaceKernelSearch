from defaultOptions import default_options, hyperparameter_limits as hpl, prior_dict as pd

global options
options = default_options
global hyperparameter_limits
hyperparameter_limits = hpl
global prior_dict
prior_dict = pd

def init():
    """
    Initialize global parameters for options (dict containing information for multiple parts of the pipeline, see defaultOptions)
    and hyperparameter_limits (dict containing ranges to search in for hyperparameter optimization, see defaultOptions)

    This needs to be called before any training!
    """
    global options, hyperparameter_limits, prior_dict
    options = default_options
    hyperparameter_limits = hpl
    prior_dict = pd
