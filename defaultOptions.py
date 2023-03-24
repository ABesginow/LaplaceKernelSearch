"""
This script provides all custom default options used in different steps of GP operations,
as well as operations to fill an arbitrary dictionary of options with the missing default values.
"""

default_options = {
    "training": {"print_training_output" : False,
                 "print_optimizing_output": False,
                 "max_iter": 50,
                 "learning_rate": 0.1,
                 "restarts": 5,
                 "optimization method": "default"},
    "kernel search": {"print": True,
                      "probability graph": False,
                      "multithreading": False},
    "plotting": {"border_ratio": 0.0,
                 "sample_points": 1000,
                 "legend": True}}


hyperparameter_limits = {"RBFKernel": {"lengthscale": [1e-3,1]},
                         "LinearKernel": {"variance": [1e-4,1]},
                         "PeriodicKernel": {"lengthscale": [1e-3,1],
                                            "period_length": [1e-3,3]},
                         "ScaleKernel": {"outputscale": [1e-3,10]},
                         "Noise": [1e-2,1e-1]}

prior_dict = {"SE": {"raw_lengthscale" : {"mean": 0.891, "std":2.195}},
              "PER":{"raw_lengthscale": {"mean": 0.338, "std":2.636}, "raw_period_length":{"mean": 0.284, "std":0.902}},
              "LIN":{"raw_variance" : {"mean":-1.463, "std":1.633}},
              "c":{"raw_outputscale": {"mean":-2.163, "std":2.448}},
              "noise": {"raw_noise": {"mean":-1.792, "std":3.266}}}
