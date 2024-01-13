import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from numba import jit, njit, float32, float64, uint, vectorize

np.seterr(all='warn')
import warnings

class StateSpaceModel:
    def __init__(self, m, p, q_init, a_init):
        self.m = m  # number of sources
        self.p = p  # order
        self.q = q_init  # initial state-noise covariance matrix
        self.a = a_init  # initial state-transition matrix
