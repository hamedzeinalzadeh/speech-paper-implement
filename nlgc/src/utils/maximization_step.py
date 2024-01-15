import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from numba import jit, njit, float32, float64, uint, vectorize
import itertools

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

        def solve_for_state_transition_matrix(self, lambda2, max_iterations=5000, tolerance=1e-3, zeroed_index=None, update_only_target=False, num_eigenmodes=1):
            """
            Solves for the state-transition matrix 'a' using gradient descent.

            Parameters:
            lambda2 (float): Regularization parameter.
            max_iterations (int): Maximum number of iterations for gradient descent, default is 5000.
            tolerance (float): Convergence tolerance, default is 1e-3.
            zeroed_index (tuple or None): Indices to be zeroed in the state-transition matrix. Defaults to None.
            update_only_target (bool): Flag to update only the target indices if zeroed_index is specified. Defaults to False.
            num_eigenmodes (int): Number of eigenmodes to consider. Defaults to 1.

            Returns:
            Tuple: (Updated state-transition matrix, changes across iterations)
            """

            # Check if targeted update is required or not
            if not update_only_target or zeroed_index is None:
                # Perform full update
                return self._internal_solve_for_a(self.state_noise_covariance, self.s1, self.s2, self.state_transition_matrix, self.order, lambda2, max_iterations, tolerance, zeroed_index, num_eigenmodes)
            else:
                # Perform targeted update
                target_indices = np.unique(zeroed_index[0])
                reduced_s1 = self.s1[target_indices]
                reduced_a = self.state_transition_matrix[target_indices]
                reduced_q = self.state_noise_covariance[target_indices, target_indices]
                if reduced_q.ndim == 1:
                    reduced_q = reduced_q[:, None]
                adjusted_target_indices = np.asarray(zeroed_index[0]) - min(zeroed_index[0])
                source_indices = np.asarray(zeroed_index[1])

                reduced_a, changes = self._internal_solve_for_a(reduced_q, reduced_s1, self.s2, reduced_a, self.order, lambda2, max_iterations, tolerance, (adjusted_target_indices, source_indices), num_eigenmodes)
                self.state_transition_matrix[target_indices] = reduced_a
                return self.state_transition_matrix, changes

        def _optimize_state_transition_matrix(self, lambda2, max_iterations=5000, tolerance=1e-3, zeroed_index=None, num_eigenmodes=1):
            """
            Optimizes the state-transition matrix 'a' using gradient descent.

            Parameters:
            lambda2 (float): Regularization parameter.
            max_iterations (int): Maximum number of gradient descent iterations.
            tolerance (float): Convergence tolerance.
            zeroed_index (tuple or None): Indices in 'a' to be forced to zero. Defaults to None.
            num_eigenmodes (int): Number of eigenmodes used in optimization. Defaults to 1.

            Returns:
            Tuple: (Optimized state-transition matrix, list of changes per iteration)
            """
            if lambda2 == 0:
                try:
                    self.state_transition_matrix = linalg.solve(self.s2, self.s1.T, assume_a='pos').T
                except linalg.LinAlgError:
                    self.state_transition_matrix = linalg.solve(self.s2, self.s1.T, assume_a='sym').T
                return self.state_transition_matrix, None

            # Initialize variables and parameters for optimization
            q_diagonal = np.diag(self.state_noise_covariance)
            q_inv_sqrt = 1 / np.sqrt(q_diagonal)
            scaling_factors = np.sqrt(np.diag(self.s2))
            normalized_s2 = self._normalize_matrix(self.s2, scaling_factors)
            normalized_s1 = self.s1 / scaling_factors[None, :]
            scaled_a = self.state_transition_matrix * scaling_factors[None, :]

            scaled_a *= q_inv_sqrt
            normalized_s1 *= q_inv_sqrt

            h_norm = np.linalg.eigvalsh(normalized_s2).max()
            tau_max = 0.99 / h_norm

            changes = np.zeros(max_iterations + 1)
            changes[0] = 1
            function_values = np.zeros(max_iterations + 1)
            prev_a = np.empty_like(scaled_a)
            temp_state_transition = np.empty_like(scaled_a)
            num_sources = scaled_a.shape[0]
            order = scaled_a.shape[1] // num_sources

            # Compute initial function value
            function_value_old = self._compute_function_value(scaled_a, normalized_s1, normalized_s2)
            function_values[0] = function_value_old
            temp_product = scaled_a.dot(normalized_s2)

            # Gradient descent iterations
            for iteration in range(max_iterations):
                if changes[iteration] < tolerance:
                    break

                prev_a[:] = scaled_a
                gradient = 2 * (temp_product - normalized_s1)

                # Determine optimal step size
                try:
                    step_size = self._compute_optimal_step_size(gradient, normalized_s2, tau_max)
                except RuntimeError as e:
                    raise e

                # Gradient descent step
                scaled_a, temp_product, function_value_new = self._gradient_descent_step(prev_a, gradient, step_size, lambda2, num_sources, order, p1, zeroed_index, num_eigenmodes)

                # Update function value and check for convergence
                changes[iteration + 1], function_values[iteration + 1] = self._update_convergence_metrics(prev_a, scaled_a, function_value_old, function_value_new)
                function_value_old = function_value_new

            # De-normalize the optimized state-transition matrix
            self.state_transition_matrix = scaled_a / q_inv_sqrt / scaling_factors[None, :]

            return self.state_transition_matrix, changes

        @staticmethod
        @vectorize([float32(float32, float32), float64(float64, float64)], cache=True)
        def soft_thresholding(value, threshold):
            """
            Applies soft thresholding to the given value. It's a common operation in regularization techniques.

            Parameters:
            value (float): The value to apply the thresholding to.
            threshold (float): The threshold value.

            Returns:
            float: The result after applying soft thresholding.
            """
            if value > threshold:
                return value - threshold
            elif value < -threshold:
                return value + threshold
            else:
                return 0.0

        @staticmethod
        @njit([float32[:,:](float32[:,:], uint), float64[:,:](float64[:,:], uint)], cache=True)
        def process_matrix_with_eigenmodes(matrix, num_eigenmodes):
            """
            Processes the matrix with respect to eigenmodes, aggregating values across specific rows and resetting others.

            Parameters:
            matrix (ndarray): The input matrix to be processed.
            num_eigenmodes (uint): The number of eigenmodes to consider in the processing.

            Returns:
            ndarray: The processed matrix.
            """
            processed_matrix = np.empty_like(matrix)
            num_rows, num_columns = matrix.shape
            stride = num_rows // num_eigenmodes

            for row in range(num_rows):
                eigenmode_block_start = (row // num_eigenmodes) * num_eigenmodes
                is_within_eigenmode_block = eigenmode_block_start <= row < eigenmode_block_start + num_eigenmodes

                for column_start in range(0, num_columns, num_rows):
                    column_end = column_start + stride
                    if is_within_eigenmode_block:
                        # Reset values outside the current eigenmode block
                        processed_matrix[row, column_start:column_end] = 0.0
                        # Preserve the value within the eigenmode block
                        processed_matrix[row, column_start + row] = matrix[row, column_start + row]
                    else:
                        # Aggregate values across the eigenmode block
                        processed_matrix[row, column_start:column_end] = matrix[row, column_start:column_end].sum()

            return processed_matrix

        def update_state_noise_covariance(self, lambda2, alpha=0, beta=0):
            """
            Updates the state-noise covariance matrix 'q' in the state-space model.

            Parameters:
            lambda2 (float): Regularization parameter.
            alpha (float): Parameter for the Inverse-Gamma prior, default is 0.
            beta (float): Parameter for the Inverse-Gamma prior, default is 0.

            Returns:
            Tuple: (Updated state-noise covariance matrix 'q', relative change in 'q')

            Notes:
            Non-zero alpha, beta values impose an Inverse-Gamma(alpha*n/2 - 1, beta*n) prior on 'q's.
            This is equivalent to alpha*n - 2 additional observations summing to beta*n.
            """
            diag_indices = np.diag_indices_from(self.q)
            previous_q_diag = self.q[diag_indices]

            # Compute terms used for updating 'q'
            term_a_s2 = np.einsum('ij,ji->i', self.a, self.s2.T)
            term_s2_a_s3 = np.einsum('ij,ji->i', self.s2 - self.a.dot(self.s3), self.a.T)
            updated_q_diag = self.s1[diag_indices] if self.s1.ndim == 2 else self.s1

            # Updating diagonal elements of 'q'
            updated_q_diag -= (term_a_s2 + term_s2_a_s3)
            updated_q_diag += beta
            updated_q_diag /= (1 + alpha)
            self.q[diag_indices] = np.abs(updated_q_diag)

            # Calculating relative change in 'q'
            relative_change_in_q = np.square(previous_q_diag - updated_q_diag).sum() / np.square(previous_q_diag).sum()

            return self.q, relative_change_in_q

    def compute_cross_log_likelihood(self):
        """
        Computes the cross log-likelihood for the state-space model.

        Returns:
        float: The computed log-likelihood value.

        Notes:
        This log-likelihood value is not normalized by the number of time samples (T).
        """
        # Extracting the first 'm' components from the state estimates
        estimated_states = self.x_[:, :self.num_sources]

        # Adjusting for the model order
        adjusted_sample_size = self.x_.shape[0] - self.model_order

        # Difference between actual and predicted states
        state_diff = estimated_states[self.model_order:] - self.x_[self.model_order - 1:-1].dot(self.state_transition_matrix.T)

        # Compute the log-likelihood value
        cross_log_likelihood = -0.5 * np.sum(np.square(state_diff))

        return cross_log_likelihood

    def compute_log_likelihood(self):
        """
        Computes the log-likelihood for the state-space model in a single method.

        Returns:
        float: The computed log-likelihood value.

        Notes:
        This log-likelihood value is not normalized by the number of time samples (T).
        """
        num_samples = self.x_.shape[0]
        adjusted_sample_size = num_samples - self.model_order

        # Compute log-likelihood contributions from q, r, and s_hat directly
        log_likelihood = 0

        # Contribution from the state noise covariance q
        if self.q.ndim == 1:
            self.q = np.diag(self.q)
        cholesky_q = linalg.cholesky(self.q, lower=True)
        q_cholesky_inv = linalg.solve_triangular(cholesky_q, np.eye(
            self.num_sources), lower=True, check_finite=False)
        state_diff = self.estimated_states[self.model_order:] - self.x_[
            self.model_order - 1:-1].dot(self.state_transition_matrix.T)
        log_likelihood += adjusted_sample_size * \
            np.log(np.diag(q_cholesky_inv)).sum() - 0.5 * \
            np.sum(np.square(state_diff.dot(q_cholesky_inv.T)))

        # Contribution from the observation noise covariance r
        cholesky_r = linalg.cholesky(self.r, lower=True)
        r_cholesky_inv = linalg.solve_triangular(cholesky_r, np.eye(
            self.num_channels), lower=True, check_finite=False)
        observation_diff = self.y[self.model_order:] - \
            self.estimated_states[self.model_order:].dot(
                self.observation_matrix.T)
        log_likelihood += adjusted_sample_size * np.log(np.diag(r_cholesky_inv)).sum(
        ) - 0.5 * np.sum(np.square(observation_diff.dot(r_cholesky_inv.T)))

        # Contribution from the smooth covariance matrix s_hat
        cholesky_s_hat = linalg.cholesky(
            self.s_hat[self.model_order - 1:, self.model_order - 1:], lower=True)
        s_hat_cholesky_inv = linalg.solve_triangular(
            cholesky_s_hat, np.eye(self.num_sources), lower=True, check_finite=False)
        log_likelihood += adjusted_sample_size * \
            np.log(np.diag(s_hat_cholesky_inv)).sum()

        return log_likelihood

    def compute_observation_model_log_likelihood(self):
        """
        Computes the log-likelihood component related to the observation model in the state-space framework.

        Returns:
        float: The computed log-likelihood value for the observation model.

        Notes:
        This log-likelihood value is not normalized by the number of time samples (T).
        """
        # Extract the relevant portion of the state estimate
        estimated_states = self.x_[:, :self.num_sources]

        # Compute the difference between the observed and predicted observations
        observation_diff = self.y[self.model_order:] - \
            estimated_states[self.model_order:].dot(self.observation_matrix.T)

        # Calculate the log-likelihood value for the observation model
        log_likelihood_value = -np.sum(np.square(observation_diff))

        return log_likelihood_value


if __name__ == "__main__":
    # Define parameters
    num_channels, num_sources, model_order, num_iterations = 3, 3, 2, 100
    lambda2 = 0.1
    tolerance = 1e-8
    max_iter = 1000

    # Initialize matrices
    q_matrix = np.eye(num_sources)
    r_matrix = np.eye(num_channels)
    a_matrix = np.zeros(
        (model_order, num_sources, num_sources), dtype=np.float64)
    a_matrix[1, 0, 1] = -0.2
    a_matrix[0, 0, 0] = 0.9
    a_matrix[0, 1, 1] = 0.9

    # Generate random signal and noise
    signal = np.random.standard_normal(
        (num_sources + num_channels) * num_iterations)
    state_noise = signal[:num_sources *
                         num_iterations].reshape((num_iterations, num_sources))
    l = linalg.cholesky(q_matrix, lower=True)
    state_noise = state_noise.dot(l.T)
    observation_noise = signal[num_sources *
                               num_iterations:].reshape((num_iterations, num_channels))
    l = linalg.cholesky(r_matrix, lower=True)
    observation_noise = observation_noise.dot(l.T)

    # Observation matrix
    observation_matrix = np.random.randn(num_channels, num_sources)

    # Generate states and observations
    states = np.empty((num_iterations, num_sources), dtype=np.float64)
    states[0], states[1] = 0.0, 0.0
    for i in range(2, num_iterations):
        states[i] = a_matrix[0].dot(
            states[i-1]) + a_matrix[1].dot(states[i-2]) + state_noise[i]

    observations = states.dot(observation_matrix.T) + observation_noise

    # Prepare data for solving a and q
    extended_states = np.empty(
        (num_iterations - 1, num_sources * model_order), dtype=np.float64)
    for i in range(1, num_iterations):
        extended_states[i-1, :num_sources] = states[i]
        extended_states[i-1, num_sources:] = states[i-1]

    s1 = states[2:].T.dot(extended_states[:-1]) / \
        (extended_states.shape[0] - model_order)
    s2 = extended_states[:-1].T.dot(extended_states[:-1]) / \
        (extended_states.shape[0] - model_order)
    s3 = states[2:].T.dot(states[2:]) / \
        (extended_states.shape[0] - model_order)

    # Instantiate StateSpaceModel
    model = StateSpaceModel()

    # Solve for state transition matrix and state noise covariance
    for _ in range(num_iterations):
    a_ = model.solve_for_state_transition_matrix(lambda2=lambda2, max_iterations=max_iter, tolerance=tolerance)
    q_ = model.update_state_noise_covariance(lambda2=lambda2)

    # Testing with zeroed indices
    for i, j in itertools.product(range(num_sources), repeat=2):
        if i == j:
            continue
        for _ in range(num_iterations):
            model.zeroed_index = [(i, i), (j, j + model_order)]
            a_ = model.solve_for_state_transition_matrix(lambda2=lambda2, max_iterations=max_iter, tolerance=tolerance, zeroed_index=model.zeroed_index)
            q_ = model.update_state_noise_covariance(lambda2=lambda2)

    # Visualize states and observations
    plt.figure()
    plt.plot(states, label='States')
    plt.plot(observations, label='Observations', linestyle='--')
    plt.legend()
    plt.title("States and Observations")
    plt.show()