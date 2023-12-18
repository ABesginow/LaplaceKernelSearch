import stan
import numpy as np
data = {'N': 5, 'D': 2, 'x': [-1.2649, -0.6325, 0.0, 0.6325, 1.2649], 'y': [0.0509, -0.2846, 0.1393, 0.3349, -0.1463], 't_mu': [-3.516406536102295, -0.21221138536930084], 't_sigma': [[12.838834762573242, 0.0], [0.0, 3.570371150970459]]}
data = {'N': 5, 'D': 2, 'x': [-1.2649110555648804, -0.6324555277824402, 0.0, 0.6324555277824402, 1.2649110555648804], 'y': [0.05094000697135925, -0.28463214635849, 0.13930504024028778, 0.3348914682865143, -0.14629033207893372], 't_mu': [-3.516406536102295, -0.21221138536930084], 't_sigma': [[12.838834762573242, 0.0], [0.0, 3.570371150970459]]}
data = {'N': 5, 'D': 2, 'x': [-1.2649, -0.6325, 0.0, 0.6325, 1.2649], 'y': [0.0509, -0.2846, 0.1393, 0.3349, -0.1463], 't_mu': [-3.516406536102295, -0.21221138536930084], 't_sigma': [[12.838834762573242, 0.0], [0.0, 3.570371150970459]]}
all_code_notebook = """
    functions {
        array[] real softplus(array[] real v){
            array[num_elements(v)] real r;
            for (d in 1:num_elements(v)){
                r[d] = log1p(exp(v[d]));
            }
            return r;
        }
        real softplus(real v){
            return log1p(exp(v));
        }
    }
    
    data {
        int N;
        int D;
        array[N] real x;
        vector[N] y;
        vector[D] t_mu;
        matrix[D, D] t_sigma;
    }
    
    parameters {
        vector<lower=-30>[D] theta;
    }
    
    model {
        matrix[N, N] K;
        vector[N] mu;
        theta ~ multi_normal(t_mu, t_sigma);
        K = (identity_matrix(dims(x)[1]).*1e-10) + (identity_matrix(dims(x)[1]).*softplus(theta[1])) + gp_exp_quad_cov(x, 1.0, softplus(theta[2]));
        mu = zeros_vector(N);
        y ~ multi_normal(mu, K);
    }
    """
all_code_metrics = """
   functions {
        array[] real softplus(array[] real v){
            array[num_elements(v)] real r;
            for (d in 1:num_elements(v)){
                r[d] = log1p(exp(v[d]));
            }
            return r;
        }
        real softplus(real v){
            return log1p(exp(v));
        }
    }
    
    data {
        int N;
        int D;
        array[N] real x;
        vector[N] y;
        vector[D] t_mu;
        matrix[D, D] t_sigma;
    }
    
    parameters {
        vector<lower=-30>[D] theta;
    }
    
    model {
        matrix[N, N] K;
        vector[N] mu;
        theta ~ multi_normal(t_mu, t_sigma);
        K = (identity_matrix(dims(x)[1]).*1e-10) + (identity_matrix(dims(x)[1]).*softplus(theta[1])) + gp_exp_quad_cov(x, 1.0, softplus(theta[2]));
        mu = zeros_vector(N);
        y ~ multi_normal(mu, K);
    }
     
"""
#data = {
#    "N": 10,
#    "D": 2,
#    "y" : [np.sin(np.pi*i) for i in range(10)],
#    "x" : [i for i in range(10)],
#    "t_mu" : [1, 1],
#    "t_sigma" : [[4, 0], [0, 4]]}
#
#with open("test.stan", "r") as f:
#    all_code = f.readlines()

code = "".join(all_code_notebook)
post = stan.build(code, data=data, random_seed=223997)
fit = post.sample(num_chains = 1, num_samples=1000)
print(fit.to_frame())
