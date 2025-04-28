import gpytorch

class DataGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_text="RBF", weights=None):
        super(DataGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        if kernel_text == "C*C*SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
        elif kernel_text == "SE":
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel_text == "C*SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_text == "LIN":
            self.covar_module = gpytorch.kernels.LinearKernel()
        elif kernel_text == "C*LIN":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        elif kernel_text == "LIN*SE":
            self.covar_module = gpytorch.kernels.LinearKernel() * gpytorch.kernels.RBFKernel()
        elif kernel_text == "LIN*PER":
            self.covar_module = gpytorch.kernels.LinearKernel() * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "SE+SE":
            self.covar_module =  gpytorch.kernels.RBFKernel() + gpytorch.kernels.RBFKernel()
        elif kernel_text == "RQ":
            self.covar_module = gpytorch.kernels.RQKernel()
        elif kernel_text == "PER":
            self.covar_module = gpytorch.kernels.PeriodicKernel()
        #elif kernel_text == "PER+SE":
        #    if weights is None:
        #        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #    else:
        #        self.covar_module = weights[0]*gpytorch.kernels.PeriodicKernel() + weights[1]*gpytorch.kernels.RBFKernel()
        elif kernel_text == "PER*SE":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.RBFKernel()
        #elif kernel_text == "PER*LIN":
        #    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        elif kernel_text == "MAT32":
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_text == "MAT32+MAT52":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) + gpytorch.kernels.MaternKernel()
        elif kernel_text == "MAT32*PER":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "MAT32+PER":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) + gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "MAT32*SE":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) * gpytorch.kernels.RBFKernel()
        elif kernel_text == "MAT32+SE":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) + gpytorch.kernels.RBFKernel()
        elif kernel_text == "MAT52":
            self.covar_module = gpytorch.kernels.MaternKernel()
        elif kernel_text == "MAT52*PER":
            self.covar_module =  gpytorch.kernels.MaternKernel() * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "MAT52+SE":
            self.covar_module =  gpytorch.kernels.MaternKernel() + gpytorch.kernels.RBFKernel()
        elif kernel_text == "SE*SE":
            self.covar_module =  gpytorch.kernels.RBFKernel() * gpytorch.kernels.RBFKernel()
        elif kernel_text == "(SE+RQ)*PER":
            self.covar_module =  (gpytorch.kernels.RBFKernel() + gpytorch.kernels.RQKernel()) * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "SE+SE+SE":
            self.covar_module =  gpytorch.kernels.RBFKernel() + gpytorch.kernels.RBFKernel() + gpytorch.kernels.RBFKernel()
        elif kernel_text == "MAT32+(MAT52*PER)":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) + (gpytorch.kernels.MaternKernel() * gpytorch.kernels.PeriodicKernel())

        #elif kernel_text == "RQ*PER":
        #    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        #elif kernel_text == "RQ*MAT32":
        #    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        #elif kernel_text == "RQ*SE":
        #    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_text="RBF", weights=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        if kernel_text == "C*C*SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
        elif kernel_text == "SE":
            self.covar_module = gpytorch.kernels.RBFKernel()
        elif kernel_text == "C*SE":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif kernel_text == "LIN":
            self.covar_module = gpytorch.kernels.LinearKernel()
        elif kernel_text == "C*LIN":
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        elif kernel_text == "LIN*SE":
            self.covar_module = gpytorch.kernels.LinearKernel() * gpytorch.kernels.RBFKernel()
        elif kernel_text == "LIN*PER":
            self.covar_module = gpytorch.kernels.LinearKernel() * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "SE+SE":
            self.covar_module =  gpytorch.kernels.RBFKernel() + gpytorch.kernels.RBFKernel()
        elif kernel_text == "RQ":
            self.covar_module = gpytorch.kernels.RQKernel()
        elif kernel_text == "PER":
            self.covar_module = gpytorch.kernels.PeriodicKernel()
        #elif kernel_text == "PER+SE":
        #    if weights is None:
        #        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #    else:
        #        self.covar_module = weights[0]*gpytorch.kernels.PeriodicKernel() + weights[1]*gpytorch.kernels.RBFKernel()
        elif kernel_text == "PER*SE":
            self.covar_module = gpytorch.kernels.PeriodicKernel() * gpytorch.kernels.RBFKernel()
        #elif kernel_text == "PER*LIN":
        #    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        elif kernel_text == "MAT32":
            self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_text == "MAT32+MAT52":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) + gpytorch.kernels.MaternKernel()
        elif kernel_text == "MAT32*PER":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "MAT32+PER":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) + gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "MAT32*SE":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) * gpytorch.kernels.RBFKernel()
        elif kernel_text == "MAT32+SE":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) + gpytorch.kernels.RBFKernel()
        elif kernel_text == "MAT52":
            self.covar_module = gpytorch.kernels.MaternKernel()
        elif kernel_text == "MAT52*PER":
            self.covar_module =  gpytorch.kernels.MaternKernel() * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "MAT52+SE":
            self.covar_module =  gpytorch.kernels.MaternKernel() + gpytorch.kernels.RBFKernel()
        elif kernel_text == "SE*SE":
            self.covar_module =  gpytorch.kernels.RBFKernel() * gpytorch.kernels.RBFKernel()
        elif kernel_text == "(SE+RQ)*PER":
            self.covar_module =  (gpytorch.kernels.RBFKernel() + gpytorch.kernels.RQKernel()) * gpytorch.kernels.PeriodicKernel()
        elif kernel_text == "SE+SE+SE":
            self.covar_module =  gpytorch.kernels.RBFKernel() + gpytorch.kernels.RBFKernel() + gpytorch.kernels.RBFKernel()
        elif kernel_text == "MAT32+(MAT52*PER)":
            self.covar_module =  gpytorch.kernels.MaternKernel(nu=1.5) + (gpytorch.kernels.MaternKernel() * gpytorch.kernels.PeriodicKernel())

        #elif kernel_text == "RQ*PER":
        #    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        #elif kernel_text == "RQ*MAT32":
        #    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        #elif kernel_text == "RQ*SE":
        #    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel()) * gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class ExactMIGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_text="SE", weights=None):
        super(ExactMIGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        if kernel_text == "[SE; SE]":
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.RBFKernel(active_dims=0) + gpytorch.kernels.RBFKernel(active_dims=1), num_dims=2)
        elif kernel_text == "[SE+SE; 1]2":
            self.covar_module = gpytorch.kernels.RBFKernel(active_dims=0) + gpytorch.kernels.RBFKernel(active_dims=0)
        elif kernel_text == "[SE+SE; 1]":
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.RBFKernel(active_dims=0) + gpytorch.kernels.RBFKernel(active_dims=0), num_dims=2)
        elif kernel_text == "[SE;* SE]":
            #TODO
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.RBFKernel(active_dims=0) + gpytorch.kernels.RBFKernel(active_dims=1), num_dims=2)
        elif kernel_text == "[SE; LIN]":
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.RBFKernel(active_dims=0) + gpytorch.kernels.LinearKernel(active_dims=1), num_dims=2)
        elif kernel_text == "[LIN; SE]":
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.LinearKernel(active_dims=0) + gpytorch.kernels.RBFKernel(active_dims=1), num_dims=2)
        elif kernel_text == "[LIN; LIN]":
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.LinearKernel(active_dims=0) + gpytorch.kernels.LinearKernel(active_dims=1), num_dims=2)
        elif kernel_text == "[LIN;* LIN]":
            self.covar_module = gpytorch.kernels.LinearKernel(active_dims=0) * gpytorch.kernels.LinearKernel(active_dims=1)
        elif kernel_text == "[SE;* 1]":
            self.covar_module = gpytorch.kernels.RBFKernel(active_dims=0)
        elif kernel_text == "[PER;* 1]":
            self.covar_module = gpytorch.kernels.PeriodicKernel(active_dims=0)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class DataMIGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_text="SE", weights=None):
        super(DataMIGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        if kernel_text == "[SE; SE]":
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.RBFKernel(active_dims=0) + gpytorch.kernels.RBFKernel(active_dims=1), num_dims=2)
        elif kernel_text == "[SE+SE; 1]2":
            self.covar_module = gpytorch.kernels.RBFKernel(active_dims=0) + gpytorch.kernels.RBFKernel(active_dims=0)
        elif kernel_text == "[SE+SE; 1]":
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.RBFKernel(active_dims=0) + gpytorch.kernels.RBFKernel(active_dims=0), num_dims=2)
        elif kernel_text == "[SE;* SE]":
            #TODO
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.RBFKernel(active_dims=0) + gpytorch.kernels.RBFKernel(active_dims=1), num_dims=2)
        elif kernel_text == "[SE; LIN]":
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.RBFKernel(active_dims=0) + gpytorch.kernels.LinearKernel(active_dims=1), num_dims=2)
        elif kernel_text == "[LIN; SE]":
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.LinearKernel(active_dims=0) + gpytorch.kernels.RBFKernel(active_dims=1), num_dims=2)
        elif kernel_text == "[LIN; LIN]":
            self.covar_module = gpytorch.kernels.AdditiveStructureKernel(gpytorch.kernels.LinearKernel(active_dims=0) + gpytorch.kernels.LinearKernel(active_dims=1), num_dims=2)
        elif kernel_text == "[LIN;* LIN]":
            self.covar_module = gpytorch.kernels.LinearKernel(active_dims=0) * gpytorch.kernels.LinearKernel(active_dims=1)
        elif kernel_text == "[SE;* 1]":
            self.covar_module = gpytorch.kernels.RBFKernel(active_dims=0)
        elif kernel_text == "[PER;* 1]":
            self.covar_module = gpytorch.kernels.PeriodicKernel(active_dims=0)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
