
import matplotlib.pyplot as plt
import numpy as np
from codetiming import Timer
from loguru import logger
from matplotlib import pyplot as plt
from scipy import linalg


class SteadyStateKalmanFilter:
    """
    This class implements a Steady-State Kalman Filter.
    """

    def __init__(self, a, f, q, r, use_lapack=True):
        """
        Initializes the Steady-State Kalman Filter with system parameters.

        Parameters:
            a (ndarray): State transition matrix of shape (n_sources*order, n_sources*order).
            f (ndarray): Measurement function matrix of shape (n_channels, n_sources*order).
            q (ndarray): Process noise covariance matrix of shape (n_sources*order, n_sources*order).
            r (ndarray): Measurement noise covariance matrix of shape (n_channels, n_channels).
            initial_states (tuple, optional): Tuple of two ndarrays for initial state (prior and posterior).
            use_lapack (bool): Use LAPACK optimizations if True.
        """
        self.a = a
        self.f = f
        self.q = q
        self.r = r
        self.use_lapack = use_lapack


    def sskf(self, y, xs=None, use_lapack=True):
        """
        Computes steady-state smoothed distribution using a Kalman filter.

        Parameters:
        xs(tuple, optional): Tuple of two ndarrays for initial state (prior and posterior states).
        use_lapack(bool): Flag to use LAPACK optimizations if True.

        Returns:
        Tuple containing state estimates, smoothed error covariances, smoothing gain, additional smoothed covariances, and negative log likelihood.
        """
        a, f, q, r = self.a, self.f, self.q, self.r  # Use instance variables
        use_lapack = self.use_lapack
        num_samples, observation_dim = y.shape
        _, state_dim = f.shape
        num_samples, observation_dim = y.shape
        _, state_dim = f.shape

        # Initialize state estimates
        prior_state, posterior_state = (np.empty(
            (num_samples, state_dim), dtype=np.float64), np.empty_like(prior_state)) if xs is None else xs

        # Solve the discrete algebraic Riccati equation for steady-state covariance
        try:
            steady_state_cov = linalg.solve_discrete_are(
                a.T, f.T, q, r, balanced=not use_lapack)
        except np.linalg.LinAlgError:
            steady_state_cov = linalg.solve_discrete_are(
                a.T, f.T, q, r, balanced=use_lapack)

        # Compute Kalman gain and prepare variables for log-likelihood computation
        temp_matrix = f.dot(steady_state_cov)
        innovation_covariance = temp_matrix.dot(f.T) + r
        cholesky_factor, lower = linalg.cho_factor(
            innovation_covariance, check_finite=False)
        kalman_gain = linalg.cho_solve(
            (cholesky_factor, lower), temp_matrix, check_finite=False).T
        inv_innov_cov = linalg.cho_solve(
            (cholesky_factor, lower), np.eye(r.shape[0]), check_finite=False)
        log_likelihood = 0.0

        # Forward pass of the Kalman filter
        for i in range(num_samples):
            prior_state[i] = np.zeros(state_dim) if i == 0 else a.dot(
                posterior_state[i - 1])
            innovation = y[i] - f.dot(prior_state[i])
            posterior_state[i] = prior_state[i] + kalman_gain.dot(innovation)
            log_likelihood += 0.5 * \
                np.sum(inv_innov_cov.dot(innovation) * innovation)

        # Compute smoother gain and smoothed error covariance
        smoother_gain_residual = steady_state_cov - kalman_gain.dot(temp_matrix)
        smoother_gain = ...  # Smoother gain computation
        smoothed_error_cov = linalg.solve_discrete_lyapunov(
            smoother_gain, smoother_gain_residual)
        additional_smoothed_cov = steady_state_cov - \
            smoother_gain.dot(steady_state_cov).dot(smoother_gain.T)

        # Backward pass for smoother
        for i in reversed(range(num_samples - 1)):
            state_diff = posterior_state[i + 1] - prior_state[i + 1]
            posterior_state[i] += smoother_gain.dot(state_diff)

        return posterior_state, smoothed_error_cov, smoother_gain, additional_smoothed_cov, -log_likelihood

    def sskfcv(self, y, initial_states=None):
        """
        Computes the cross-validated steady-state smoothed distribution.

        Parameters:
            y (ndarray): Observation array of shape (n_samples, n_channels).
            initial_states (tuple, optional): Tuple of two ndarrays for initial state.
                                              If None, zeros are used for initialization.

        Returns:
            float: The negative model fit metric.
        """
        # Ensure the observation dimension matches the measurement matrix
        assert y.shape[1] == self.f.shape[0], "Dimension mismatch in 'y' and 'f'"

        num_samples, observation_dim = y.shape
        _, state_dim = self.f.shape

        # Initialize state estimates for filtering
        if initial_states is None:
            filtered_state_previous = np.zeros((num_samples, state_dim))
            filtered_state_current = np.zeros_like(filtered_state_previous)
        else:
            filtered_state_previous, filtered_state_current = initial_states
            assert filtered_state_previous.shape == (
                num_samples, state_dim), "Invalid shape for initial states"

        # Solve the discrete algebraic Riccati equation for steady-state covariance
        try:
            steady_state_cov = linalg.solve_discrete_are(
                self.a.T, self.f.T, self.q, self.r, balanced=True)
        except ValueError:
            steady_state_cov = linalg.solve_discrete_are(
                self.a.T, self.f.T, self.q, self.r, balanced=False)

        # Compute the Kalman gain matrix
        temp_matrix = self.f.dot(steady_state_cov)
        innovation_covariance = temp_matrix.dot(self.f.T) + self.r
        cholesky_factor, lower = linalg.cho_factor(
            innovation_covariance, check_finite=False)
        kalman_gain = linalg.cho_solve(
            (cholesky_factor, lower), temp_matrix, check_finite=False).T

        # Apply the Kalman filter for each time step
        for time_step in range(num_samples):
            if time_step == 0:
                filtered_state_previous[time_step] = np.zeros(state_dim)
            else:
                filtered_state_previous[time_step] = self.a @ filtered_state_current[time_step - 1]

            observation_innovation = y[time_step] - \
                self.f @ filtered_state_previous[time_step]
            filtered_state_current[time_step] = filtered_state_previous[time_step] + \
                kalman_gain @ observation_innovation

        # Calculate the model fit metric based on residuals
        residuals = y - filtered_state_current @ self.f.T
        model_fit = np.sum(residuals**2) / (num_samples *
                                            np.sum(np.diag(innovation_covariance))**2)

        return -model_fit

    def sskf_prediction(y, a, f, q, r, initial_states=None, use_lapack=True):
        """
        Computes steady-state smoothed distribution and predictions using the Kalman filter.

        Parameters:
            y (ndarray): Observation array of shape (n_samples, n_channels).
            initial_states (tuple, optional): Tuple of two ndarrays for initial state (prior and posterior).
                                            If None, zeros are used for initialization.
            use_lapack (bool, optional): Flag to use LAPACK optimizations if True. Default is True.

        Returns:
            predictions (ndarray): The array of predicted observations based on the Kalman filter,
                                of shape (n_samples, n_channels).
        """
        num_samples, num_channels = y.shape
        _, state_dim = f.shape

        # Initialize state estimates
        if initial_states is None:
            state_prior = np.zeros((num_samples, state_dim), dtype=np.float64)
            state_posterior = np.zeros_like(state_prior)
        else:
            state_prior, state_posterior = initial_states
            assert state_prior.shape == (
                num_samples, state_dim), "Invalid shape for initial states"

        # Solve the discrete algebraic Riccati equation
        try:
            steady_state_cov = linalg.solve_discrete_are(
                a.T, f.T, q, r, balanced=not use_lapack)
        except np.linalg.LinAlgError:
            steady_state_cov = linalg.solve_discrete_are(
                a.T, f.T, q, r, balanced=use_lapack)

        # Compute the Kalman gain matrix
        temp_matrix = f.dot(steady_state_cov)
        innovation_covariance = temp_matrix.dot(f.T) + r
        cholesky_factor, lower = linalg.cho_factor(
            innovation_covariance, check_finite=False)
        kalman_gain = linalg.cho_solve(
            (cholesky_factor, lower), temp_matrix, check_finite=False).T

        # Prediction and filtering process
        predictions = np.empty_like(y)
        for time_step in range(num_samples):
            # Predict the next state
            if time_step == 0:
                state_prior[time_step] = np.zeros(state_dim)
            else:
                state_prior[time_step] = a.dot(state_posterior[time_step - 1])

            # Update the state with the observation
            observation_innovation = y[time_step] - \
                f.dot(state_prior[time_step])
            state_posterior[time_step] = state_prior[time_step] + \
                kalman_gain.dot(observation_innovation)

            # Store the prediction
            predictions[time_step] = f.dot(state_prior[time_step])

        return predictions

    def align_cast(array_tuple, use_lapack):
        """
        Internal function to typecast and/or memory-align ndarrays. This function
        ensures that all arrays in the given tuple are of type np.float64 and are
        memory-aligned as required for LAPACK optimizations.

        Parameters:
            array_tuple (tuple of ndarrays): Tuple containing arrays of arbitrary shapes.
                                            Each array in this tuple will be checked and
                                            potentially modified for dtype and memory alignment.
            use_lapack (bool): Flag indicating whether to make arrays F-contiguous
                            for LAPACK optimizations.

        Returns:
            tuple of ndarrays: A tuple where each array has been typecast to np.float64
                            and made F-contiguous if `use_lapack` is True.
        """
        typecasted_arrays = (array.astype(np.float64) if array.dtype != np.float64 else array
                             for array in array_tuple)

        if use_lapack:
            aligned_arrays = (array if array.flags['F_CONTIGUOUS'] else np.asfortranarray(array)
                              for array in typecasted_arrays)
            return tuple(aligned_arrays)

        return tuple(typecasted_arrays)


# from your_kalman_filter_module import SteadyStateKalmanFilter
# Assuming SteadyStateKalmanFilter class is defined in your_kalman_filter_module.py

if __name__ == "__main__":
    t = 1000  # Number of time steps

    # Define dimensions for state and observation
    state_dim, observation_dim = 3, 3

    # Initialize process and measurement noise covariances
    process_noise_cov = np.eye(state_dim)
    measurement_noise_cov = 0.01 * np.eye(observation_dim)

    # Generate random process and measurement noise
    random_noise = np.random.standard_normal((state_dim + observation_dim) * t)
    process_noise = np.dot(random_noise[:state_dim * t].reshape(t, state_dim),
                           linalg.cholesky(process_noise_cov, lower=True).T)
    measurement_noise = np.dot(random_noise[state_dim * t:].reshape(t, observation_dim),
                               linalg.cholesky(measurement_noise_cov, lower=True).T)

    # Initialize state transition and measurement function matrices
    _ = np.random.randn(state_dim, state_dim)
    state_transition_matrix = np.random.randn(
        state_dim, state_dim) / (1.1 * linalg.norm(_))
    measurement_function_matrix = np.eye(observation_dim)

    # Generate state and observation sequences
    state_sequence = np.empty((t, state_dim), dtype=np.float64)
    state_sequence[0] = 0.0
    for current_state, previous_state, current_noise in zip(state_sequence[1:], state_sequence, process_noise):
        current_state[:] = state_transition_matrix.dot(
            previous_state) + current_noise
    observation_sequence = state_sequence.dot(
        measurement_function_matrix.T) + measurement_noise

    # Initialize state estimates
    state_estimate_prior = np.empty_like(state_sequence)
    state_estimate_posterior = np.empty_like(state_estimate_prior)

    # Initialize timers for performance measurement
    timer_optimized = Timer(name='optimized', logger=None)
    timer_vanilla = Timer(name='vanilla', logger=None)

    # Kalman filter application with and without LAPACK optimizations
    for _ in range(10):
        with timer_optimized:
            predictions_optimized, *_ = SteadyStateKalmanFilter(state_transition_matrix, measurement_function_matrix,
                                                                process_noise_cov, measurement_noise_cov).sskf(
                observation_sequence, xs=(state_estimate_prior, state_estimate_posterior), use_lapack=True)
    logger.info(
        f"Elapsed time (optimized): {Timer.timers.mean('optimized'):.4f} ± {Timer.timers.stdev('optimized'):.4f}")

    for _ in range(10):
        with timer_vanilla:
            predictions_vanilla, *_ = SteadyStateKalmanFilter(state_transition_matrix, measurement_function_matrix,
                                                              process_noise_cov, measurement_noise_cov).sskf(
                observation_sequence, xs=(state_estimate_prior, state_estimate_posterior), use_lapack=False)

    logger.info(
        f"Elapsed time (vanilla): {Timer.timers.mean('vanilla'):.4f} ± {Timer.timers.stdev('vanilla'):.4f}")

    # Plotting the original state, optimized, and vanilla predictions
    fig, axes = plt.subplots(state_dim)
    for original_state, optimized_state, vanilla_state, ax in zip(state_sequence.T, predictions_optimized.T, predictions_vanilla.T, axes):
        ax.plot(original_state, label='Original State')
        ax.plot(optimized_state, label='Optimized Prediction')
        ax.plot(vanilla_state, label='Vanilla Prediction')
        ax.legend()
    plt.show()
