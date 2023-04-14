    functions {
        array[] real softplus(array[] real v){
            array[num_elements(v)] real r;
            for (d in 1:num_elements(v)){
                r[d] = log(1.0 + exp(v[d]));
            }
            return r;
        }
        real softplus(real v){
            return log(1.0 + exp(v));
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
        vector[D] theta;
    }

    model {
        matrix[N, N] K;
        vector[N] mu;
        theta ~ multi_normal(t_mu, t_sigma);
        K = gp_exp_quad_cov(x, theta[1], softplus(theta[2]));
        mu = zeros_vector(N);
        y ~ multi_normal(mu, K);

    }
