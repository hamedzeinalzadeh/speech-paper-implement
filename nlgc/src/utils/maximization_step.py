import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from numba import jit, njit, float32, float64, uint, vectorize

np.seterr(all='warn')
import warnings


class StateSpaceModel:
    def __init__(self, num_sources, order, initial_state_noise_cov, initial_state_transition_matrix, num_channels=None, regularization_param=0.1, max_iterations=1000, convergence_tol=1e-3):
        """
        Initializes a state space model for signal processing and system identification.

        Parameters:
        num_sources (int): The number of sources in the model, representing the dimensionality of the state vector.
        order (int): The order of the model, defining the number of previous time points to consider.
        initial_state_noise_cov (ndarray): The initial state-noise covariance matrix, representing uncertainty in the state transition.
        initial_state_transition_matrix (ndarray): The initial state-transition matrix, defining the state evolution.
        num_channels (int, optional): The number of channels in the model, applicable for multi-dimensional observations.
        regularization_param (float, optional): Regularization parameter for optimization, defaulting to 0.1.
        max_iterations (int, optional): Maximum number of iterations for optimization, defaulting to 1000.
        convergence_tol (float, optional): Convergence tolerance for optimization, defaulting to 1e-3.

        Returns:
        A StateSpaceModel object initialized with the specified parameters.
        """
        self.num_sources = num_sources
        self.order = order
        self.state_noise_covariance = initial_state_noise_cov
        self.state_transition_matrix = initial_state_transition_matrix
        self.num_channels = num_channels
        self.regularization_param = regularization_param
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tol

        def calculate_second_order_expectations(self, smoothed_means, smoothed_covariances, smoother_gain):
            """
            Performs the computation of second-order expectation values essential for state space models.

            Parameters:
            smoothed_means (ndarray): Array containing smoothed mean values, dimension (n_samples, num_sources*order).
            smoothed_covariances (ndarray): Matrix of smoothed covariance values, dimension (num_sources*order, num_sources*order).
            smoother_gain (ndarray): Matrix representing the gain from the smoother algorithm, dimension (num_sources*order, num_sources*order).

            Returns:
            A tuple (s1, s2, s3, n) with the following components:
            - s1 (ndarray): Matrix representing smoothed covariances between different time points (n, n-1), dimension (num_sources, num_sources*order).
            - s2 (ndarray): Augmented matrix of smoothed covariances (n-1, n-1), dimension (num_sources*order, num_sources*order).
            - s3 (ndarray): Smoothed covariance matrix (n, n), dimension (num_sources, num_sources).
            - n (int): Adjusted sample size post order correction.
            
            Note:
            Normalization by the number of adjusted samples optimizes the Q function for temporal analysis.
            """
            # Determine the total number of samples and adjust for the order
            total_samples = smoothed_means.shape[0]
            sample_size_post_order = total_samples - self.order

            # Derive cross-covariance from the smoother gain and initial segment of smoothed covariances
            cross_section_covariance = smoother_gain.dot(smoothed_covariances[:, :self.num_sources])
            initial_smoothed_means = smoothed_means[:, :self.num_sources]

            # Meticulous computation of second-order statistical expectations
            # s1: Smoothed covariance for current and previous time points
            s1 = initial_smoothed_means[self.order:].T.dot(smoothed_means[self.order - 1:-1]) / sample_size_post_order + cross_section_covariance.T
            # s2: Comprehensive smoothed covariance across all previous time points
            s2 = smoothed_means[self.order - 1:-1].T.dot(smoothed_means[self.order - 1:-1]) / sample_size_post_order + smoothed_covariances
            if (np.diag(s2) <= 0).any():
                raise ValueError('Non-positive values detected in diagonal of s2!')

            # s3: Condensed smoothed covariance matrix for the current time point
            s3 = initial_smoothed_means[self.order:].T.dot(initial_smoothed_means[self.order:]) / sample_size_post_order + smoothed_covariances[(self.order - 1) * self.num_sources:, (self.order - 1) * self.num_sources:]

            return s1, s2, s3, sample_size_post_order