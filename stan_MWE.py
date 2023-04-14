import stan
import numpy as np

data = {
    "N": 10,
    "D": 2,
    "y" : [np.sin(np.pi*i) for i in range(10)],
    "x" : [i for i in range(10)],
    "t_mu" : [1, 1],
    "t_sigma" : [[4, 0], [0, 4]]}

with open("test.stan", "r") as f:
    all_code = f.readlines()

code = "".join(all_code)
post = stan.build(code, data=data)
fit = post.sample(num_chains = 4, num_samples=1000)
print(fit.to_frame())
