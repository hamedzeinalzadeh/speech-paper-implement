import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from src.opt.e_step import sskf
from src.opt.m_step import solve_for_a, solve_for_q, calculate_ss, compute_ll
from src.opt.opt import NeuraLVAR, NeuraLVARCV
from src._nlgc import _gc_extraction


class ProcessGenerator:
    """
    Generates processes based on provided parameters and plots the results.

    Attributes:
        seed (int): Seed for the random number generator.
        n (int): Number of outputs.
        m (int): Number of inputs.
        p (int): Order of the autoregressive model.
        k (int): Number of non-zero entries in the autoregressive matrix.
        t (int): Length of the time series.
        n_eigenmodes (int): Number of eigenmodes.
    """

    def __init__(self, seed=0, n=3, m=3, p=2, k=4, t=1000, n_eigenmodes=2):
        self.seed = seed
        self.n = n
        self.m = m
        self.p = p
        self.k = k
        self.t = t
        self.n_eigenmodes = n_eigenmodes

    def generate(self):
        """
        Generates the processes and plots the results.

        Returns:
            tuple: Contains generated matrices x, y, f, r, p, a, q.
        """
        np.random.seed(self.seed)
        q = np.eye(self.m)
        r = 1 * np.eye(self.n)
        sn = np.random.standard_normal((self.m + self.n) * self.t)
        u, v = self._generate_uv(sn, q, r)

        a = self._generate_a()
        f, x, y = self._generate_fx(a, u, v)

        self._plot_results(x, y)
        return x, y, f, r, self.p, a, q

    def _generate_uv(self, sn, q, r):
        u = sn[:self.m * self.t].reshape((self.t, self.m))
        u = u.dot(linalg.cholesky(q, lower=True).T)

        v = sn[self.m * self.t:].reshape((self.t, self.n))
        v = v.dot(linalg.cholesky(r, lower=True).T)

        return u, v

    def _generate_a(self):
        a = np.zeros(self.p * self.m * self.m, dtype=np.float64)
        a.shape = (self.p, self.m, self.m)
        a[0, 0, 1] = -0.1
        a[0, 0, 0] = 0.98
        a[1, 1, 1] = 0.985
        a[0, 2, 2] = 0.5

        return a

    def _generate_fx(self, a, u, v):
        f = np.random.randn(self.n, self.m)
        f /= np.sqrt(f ** 2).sum(axis=0)

        x = np.zeros((self.t, self.m), dtype=np.float64)
        for i in range(2, self.t):
            x[i] = a[0].dot(x[i-1]) + a[1].dot(x[i-2]) + u[i]

        y_ = x.dot(f.T)
        y = 10 * np.sqrt((v**2).sum() / (y_**2).sum()) * y_ + v
        return f, x, y

    def _plot_results(self, x, y):
        fig, ax = plt.subplots(2)
        ax[0].plot(x)
        ax[1].plot(y)
        fig.show()


class NeuralVARTester:
    """
    Tests the NeuraLVAR and NeuraLVARCV models.

    Attributes:
        use_lapack (bool): Flag to use LAPACK linear algebra library.
        lambda2 (float): Regularization parameter.
        rel_tol (float): Relative tolerance for convergence.
        seed (int): Seed for random number generation.
    """

    def __init__(self, use_lapack=True, lambda2=0.1, rel_tol=0.0001, seed=0):
        self.use_lapack = use_lapack
        self.lambda2 = lambda2
        self.rel_tol = rel_tol
        self.seed = seed
        self.generator = ProcessGenerator(seed=self.seed)
        self.x, self.y, self.f, self.r, self.p, self.a, self.q = self.generator.generate()

    def test_neuralvar(self):
        """
        Tests the NeuraLVAR model.

        Returns:
            NeuraLVAR: The fitted NeuraLVAR model.
        """
        model = NeuraLVAR(self.p, use_lapack=self.use_lapack)
        y_train = self.y[:900]
        y_test = self.y[900:]
        model.fit(y_train.T, self.f, self.r, lambda2=self.lambda2, max_iter=200,
                  max_cyclic_iter=2, a_init=self.a.copy()*0, q_init=0.1*self.q, rel_tol=self.rel_tol)
        return model

    # Additional methods to test other functionalities can be added here.


if __name__ == '__main__':
    tester = NeuralVARTester()
    model = tester.test_neuralvar()
    # Additional code to utilize the model or perform other tests.
