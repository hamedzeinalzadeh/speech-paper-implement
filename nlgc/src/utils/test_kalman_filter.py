import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from nlgc.opt.e_step import sskf
from nlgc.opt.m_step import solve_for_a, solve_for_q, calculate_ss, compute_ll
from nlgc.opt.opt import NeuraLVAR, NeuraLVARCV
from nlgc._nlgc import _gc_extraction


class TimeSeriesModel:
    def __init__(self, seed=0, use_lapack=True, t=1000, n=3, m=3, p=2, k=4):
        self.seed = seed
        self.use_lapack = use_lapack
        self.t = t
        self.n, self.m, self.p, self.k = n, m, p, k
        self.q = np.eye(m)
        self.r = np.eye(n)
        self.setup()

    def setup(self):
        np.random.seed(self.seed)
        self.a, self.f = self.initialize_parameters()
        self.u, self.v = self.generate_noise()
        self.x, self.y = self.generate_time_series()

    def initialize_parameters(self):
        a = np.zeros((self.p, self.m, self.m), dtype=np.float64)
        a[0, 0, 1] = -0.1
        a[0, 0, 0] = 0.98
        a[1, 1, 1] = 0.985
        a[0, 2, 2] = 0.5

        f = np.random.randn(self.n, self.m)
        f /= np.sqrt(np.sum(f ** 2, axis=0))
        return a, f

    def generate_noise(self):
        sn = np.random.standard_normal((self.m + self.n) * self.t)
        u = sn[:self.m * self.t].reshape((self.t, self.m))
        v = sn[self.m * self.t:].reshape((self.t, self.n))
        u = u.dot(linalg.cholesky(self.q, lower=True).T)
        v = v.dot(linalg.cholesky(self.r, lower=True).T)
        return u, v

    def generate_time_series(self):
        x = np.zeros((self.t, self.m), dtype=np.float64)
        for i in range(2, self.t):
            x[i] = self.a[0].dot(x[i-1]) + self.a[1].dot(x[i-2]) + self.u[i]

        y_ = x.dot(self.f.T)
        y = 10 * np.sqrt(np.sum(self.v**2) / np.sum(y_**2)) * y_ + self.v
        return x, y

    def fit_neuralvar(self, lambda2=0.1, rel_tol=0.0001):
        y_train = self.y[:900]
        y_test = self.y[900:]
        model = NeuraLVAR(self.p, use_lapack=self.use_lapack)
        model = model.fit(y_train.T, self.f, self.r, lambda2=lambda2, max_iter=200, max_cyclic_iter=2,
                          a_init=self.a.copy()*0, q_init=0.1*self.q, rel_tol=rel_tol)
        return model

    # Additional methods for test_neuralvarcv, test_nlgc, etc., can be similarly defined here.

    def plot_data(self):
        fig, ax = plt.subplots(2)
        ax[0].plot(self.x)
        ax[1].plot(self.y)
        plt.show()


if __name__ == '__main__':
    # Example usage
    ts_model = TimeSeriesModel(seed=0, use_lapack=True)
    ts_model.plot_data()
    neuralvar_model = ts_model.fit_neuralvar()
    # More example calls and tests can be added here
