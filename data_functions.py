from helpFunctions import get_string_representation_of_kernel as gsr
from helpFunctions import get_full_kernels_in_kernel_expression
from globalParams import options
import gpytorch
import itertools
import numpy as np
import pandas as pd
import torch
from typing import List, Union, Callable
import types
import gp_classes

# Registry for input patterns
INPUT_PATTERNS = {}

# Registry for label patterns
LABEL_PATTERNS = {}


def register_input_pattern(name: str):
    """
    Decorator to register input patterns.
    """
    def wrapper(func):
        INPUT_PATTERNS[name] = func
        return func
    return wrapper


def register_label_pattern(name: str):
    """
    Decorator to register label patterns.
    """
    def wrapper(func):
        LABEL_PATTERNS[name] = func
        return func
    return wrapper


def list_available_patterns():
    return {
        "input_patterns": list(INPUT_PATTERNS.keys()),
        "label_patterns": list(LABEL_PATTERNS.keys())
    }

# -------------------------------------------------------
# Registering input patterns
# -------------------------------------------------------


@register_input_pattern("linear")
def linear(n_points: int, n_dim: int, **kwargs) -> torch.tensor:
    """
    Linear input pattern

    Parameters
    ----------
    n_points : int
        Number of points to generate
    n_dim : int
        Number of dimensions
    kwargs : dict
        Additional arguments
            - START : float
                Start value for the linear pattern
            - END : float
                End value for the linear pattern
            - dim_weights : list of floats
                Weights for each dimension (default: [1.0] * n_dim)

    Returns
    -------
    torch.tensor
        Generated input pattern

    """
    START = kwargs["START"] if "START" in kwargs else 0.0
    END = kwargs["END"] if "END" in kwargs else 1.0
    # Purpose of dim weighting is to have dimensions grow at different rates
    dim_weights = kwargs["dim_weights"] if "dim_weights" in kwargs else [1.0] * n_dim
    # Example: [1.0, 2.0] means that the first dimension grows at 1x and the second at 2x
    base_data = torch.stack([torch.linspace(START, END, n_points) for _ in range(n_dim)], dim=-1)
    if n_dim > 1:
        # Apply dimension weights
        for i in range(n_dim):
            base_data[:, i] = base_data[:, i] * dim_weights[i]
    return base_data


# -------------------------------------------------------
# Registering label patterns
# -------------------------------------------------------

@register_label_pattern("periodic_1D")
def periodic_1D(X):
    """
    $\\sin(x_0)$
    """
    return torch.sin(X[:,0])

@register_label_pattern("periodic_2D")
def periodic_2D(X):
    """
    $\\sin(x_0) \\cdot \\sin(x_1)$
    """
    return torch.sin(X[:,0]) * torch.sin(X[:,1])

@register_label_pattern("parabola_1D")
def parabola_1D(X):
    """
    $x_0^2$
    """
    return X[:,0]**2

@register_label_pattern("parabola_2D")
def parabola_2D(X):
    """
    $x_0^2 \\cdot x_1^2$
    """
    return X[:,0]**2 + X[:,1]**2

@register_label_pattern("product")
def product(X):
    """
    $x_0 \\cdot x_1$
    """
    return X[:,0] * X[:,1]

@register_label_pattern("periodic_sum")
def periodic_sum(X):
    """
    $\\sin(x_0 + x_1)$
    """
    return torch.sin(X[:,0] + X[:,1])

@register_label_pattern("periodic_sincos")
def periodic_sincos(X):
    """
    $\\sin(x_0) \\cdot \\cos(x_1)$
    """
    return torch.sin(X[:,0]) * torch.cos(X[:,1])


@register_label_pattern("linear_1D")
def linear_1D(X):
    """
    $x_0$
    """
    return X[:,0]

@register_label_pattern("linear_2D")
def linear_2D(X):
    """
    $x_0 + x_1$
    """
    return X[:,0]+X[:,1]



class Transformations:
    """
    Transformations are functions that can be applied to the inputs or labels.
    """

    def __init__(self):
        pass

    @staticmethod
    def z_score(x: torch.tensor, **kwargs) -> torch.tensor:
        return_factors = kwargs["return_factors"] if "return_factors" in kwargs else False
        if return_factors:
            mean = x.mean()
            std = x.std()
            return (x - mean) / std, mean, std
        else:
            # Standardize the data
            return (x - x.mean()) / x.std()

    @staticmethod
    def inverse_z_score(x: torch.tensor, mean: torch.tensor, std: torch.tensor) -> torch.tensor:
        return x * std + mean


class LabelGenerator:
    """
    Gaussian Process patterns begin with "GP_".
    """

    def __init__(self, pattern : Union[str, Callable]):
        self.pattern = pattern
        pass


    def generate_labels(self, inputs : torch.tensor) -> torch.tensor:
        if callable(self.pattern):
            return self.pattern(inputs)
        else:
            if self.pattern.startswith("GP_"):
                # Generate a Gaussian Process pattern
                callable_pattern = self.generate_gp_callable_pattern(self.pattern[3:], inputs)
            else:
                # Generate a standard pattern
                callable_pattern = LABEL_PATTERNS[self.pattern]
            return callable_pattern(inputs)

    
    def generate_gp_callable_pattern(self, pattern : str, inputs : torch.tensor) -> Callable:
        n_dim = inputs.shape[1]
        alibi_x_points = torch.stack([torch.linspace(0, 1, 1) for _ in range(n_dim)], dim=-1)
        alibi_y_points = torch.linspace(0, 1, 1)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if n_dim == 1:
            gp = gp_classes.DataGPModel(alibi_x_points, alibi_y_points,likelihood, kernel_text=pattern)
        elif n_dim == 2:
            gp = gp_classes.DataMIGPModel(alibi_x_points, alibi_y_points,likelihood, kernel_text=pattern)

        def gp_callable(inputs):
            gp.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.prior_mode(True):
                gp_output = gp(inputs)
                labels = gp_output.mean
                return labels
        return gp_callable





def load_csv_tensor(filepath: str, target_dim=None, header=None, expected_ndim: int = 1) -> torch.Tensor:
    data = torch.tensor(pd.read_csv(filepath, header=header).values)
    
    if target_dim is not None:
        if isinstance(target_dim, list):
            data = data[:, target_dim]
        else:
            data = data[:, target_dim]
    elif data.ndim > expected_ndim:
        data = data[:, 0]
        print(f"Warning: data has more than {expected_ndim} dimensions. Using the first dimension.")

    return data


class DataGenerator:

    def __init__(self):
        pass
    
    def generate_inputs(self, pattern: Union[str, Callable], n_points: int = 0, n_dim: int = 1, **kwargs) -> torch.tensor:
        """
        Pattern might be read from a file or be a lambda expression
        """
        if isinstance(pattern, str):
            if pattern.endswith(".csv"):
                target_dim = kwargs["target_dim"] if "target_dim" in kwargs else None
                header = kwargs["header"] if "header" in kwargs else None
                input = load_csv_tensor(pattern, target_dim=target_dim, header=header, expected_ndim=n_dim)
            else:
                standard_pattern = INPUT_PATTERNS[pattern]
                input = standard_pattern(n_points, n_dim, **kwargs)

        return input


    def generate_labels(self, inputs : torch.tensor=None,  pattern : Union[str, Callable]=None, **kwargs) -> torch.tensor:
        if isinstance(pattern, str):
            if pattern.endswith(".csv"):
                target_dim = kwargs["target_dim"] if "target_dim" in kwargs else None
                header = kwargs["header"] if "header" in kwargs else None
                n_dim = kwargs["n_dim"] if "n_dim" in kwargs else 1
                labels = load_csv_tensor(pattern, target_dim=target_dim, header=header, expected_ndim=n_dim)
            else:
                pattern = LabelGenerator(pattern)
                labels = pattern.generate_labels(inputs)
        else:
            # If pattern is a function, it should be callable
            labels = pattern(inputs)

        return labels


    def apply_transformations(self, inputs: torch.tensor, transformations: List[Callable] = None) -> torch.tensor:
        if not transformations:
            return inputs
        for transformation in transformations:
            inputs = transformation(inputs)
        return inputs
