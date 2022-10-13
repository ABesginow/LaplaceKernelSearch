"""
This script provides all custom default options used in different steps of GP operations,
as well as operations to fill an arbitrary dictionary of options with the missing default values.
"""

default_options = {
    "training": {"print_training_output" : False,
                 "print_optimizing_output": False,
                 "max_iter": 10,
                 "learning_rate": 0.1,
                 "restarts": 5,
                 "optimization method": "botorch"},
    "kernel search": {"print": False,
                      "probability graph": False,
                      "multithreading": False},
    "plotting": {"border_ratio": 0.0,
                 "sample_points": 1000,
                 "legend": True},
    "data generation": {"const": {},
                        "lin": {"a": 1},
                        "per": {"p": 1, "a": 1, "p_offset": 0}}}


hyperparameter_limits = {"RBFKernel": {"lengthscale": [1e-4,1]},
                         "LinearKernel": {"variance": [1e-4,1]},
                         "PeriodicKernel": {"lengthscale": [1e-4,10],
                                            "period_length": [1e-4,10]},
                         "ScaleKernel": {"outputscale": [1e-4,100]},
                         "WhiteNoiseKernel": {'lengthscale': [1e-4,1]},
                         "CosineKernel": {"period_length": [1e-4,10]},
                         "Noise": [1e-4,2e-3],
                         "Mean": [0.0,1.0]}

