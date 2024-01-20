import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from src.opt.e_step import sskf
from src.opt.m_step import solve_for_a, solve_for_q, calculate_ss, compute_ll
from src.opt.opt import NeuraLVAR, NeuraLVARCV
from src._nlgc import _gc_extraction


class ProcessGenerator:
    def __init__(self, seed=0):
        self.seed = seed
        # other initializations

    def generate(self):
        # logic from generate_processes
        return x, y, f, r, p, a, q

class NeuralVARModel:
    def __init__(self, p, use_lapack=True):
        # initialization

    def fit(self, y_train, f, r, lambda2, max_iter, max_cyclic_iter, a_init, q_init, rel_tol):
        # fitting logic

# Similar structure for NeuralVARCVModel and NLGC

class ExperimentRunner:
    def __init__(self):
        # initialization

    def run(self):
        # orchestrate the running of experiments

if __name__ == '__main__':
    runner = ExperimentRunner()
    runner.run()