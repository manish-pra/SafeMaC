import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import gpytorch

# We will use the simplest form of GP model, exact inference


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def d_func(x, y):
    return np.abs((x[0] - y[0]))


def get_nearest_agent(v, S):
    agent_index = 10
    min_dist = np.inf
    for k in S:
        temp = d_func(v, k)
        if temp < min_dist:
            min_dist = temp
            agent_index = k
    return agent_index, min_dist


class GridWorld:
    def __init__(self, env_dim, N) -> None:
        self.env_dim = env_dim
        self.N = N
        self.Y = self.generate_pdf()

    def generate_pdf(self):
        if self.env_dim == 1:
            pos = [x for x in range(self.N)]
            pos = np.array(pos).reshape(self.N, -1, self.env_dim)

            rv = multivariate_normal([30], [[50]])
            rv2 = multivariate_normal([80], [[50]])

            W = rv.pdf(pos) * 100 + rv2.pdf(pos) * 100
        else:
            pos = [(x, y) for x in range(self.N) for y in range(self.N)]
            pos = np.array(pos).reshape(self.N, -1, self.env_dim)

            rv = multivariate_normal([5, 2], [[20, 0.03], [0.03, 50]])
            rv2 = multivariate_normal([2, 7], [[2, 0.03], [0.03, 5]])

            # W = rv.pdf(pos)*100
            W = rv.pdf(pos) * 100 + rv2.pdf(pos) * 100

        Y = W  # May be latter I can add some noise to this.
        return Y

    def get_density_at(self, loc):
        if self.Y.ndim == 1:
            return self.Y[loc[0]]
        else:
            return self.Y[loc[0], loc[1]]

    def get_density_all(self, locs):
        density = []
        for loc in locs:
            density.append(self.get_density_at(loc))
        return density

    def plot_pdf(self):
        fig = plt.figure()
        if self.env_dim == 1:
            plt.plot(np.arange(0, self.N), self.Y)
            plt.xlabel("Position")
            plt.ylabel("Concentration")
        else:
            ax = plt.axes(projection="3d")
            ax.contour3D(np.arange(0, self.N), np.arange(0, self.N), self.Y, 300)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Obj")
        plt.show()
