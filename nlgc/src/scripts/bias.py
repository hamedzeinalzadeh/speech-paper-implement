import warnings

import numpy as np
from scipy import linalg

from src.utils.maximization_step import StateSpaceModel


def compute_deviance_bias(Q_matrix, A_matrix, X_mean, null_indices, eigenmode_count):
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

    time_steps, source_mode_dim = X_mean.shape
    _, total_dim = A_matrix.shape
    order = total_dim // source_mode_dim

    bias_total = 0
    Q_diagonal = np.diag(Q_matrix)
    cross_terms = np.zeros((time_steps - order, total_dim))

    for source_idx in range(source_mode_dim):
        A_i = A_matrix[source_idx]
        X_i = X_mean[order:, source_idx]

        # Construct cross-terms matrix for regression
        for offset in range(order):
            start_col = offset * source_mode_dim
            end_col = (offset + 1) * source_mode_dim
            cross_terms[:, start_col:end_col] = X_mean[order -
                                                       1 - offset:time_steps - 1 - offset]

        # Compute gradient and Hessian of the log-likelihood
        log_likelihood_gradient = cross_terms.T.dot(
            X_i - cross_terms.dot(A_i)) / Q_diagonal[source_idx]
        log_likelihood_hessian = - \
            cross_terms.T.dot(cross_terms) / Q_diagonal[source_idx]

        # Remove specific indices for cross-talk components
        for eigenmode_start in range(0, source_mode_dim, eigenmode_count):
            for eigenmode_u in range(eigenmode_count):
                for eigenmode_v in range(eigenmode_count):
                    if eigenmode_v != eigenmode_u and source_idx == eigenmode_start + eigenmode_v:
                        remove_idx = list(
                            range(eigenmode_start + eigenmode_u, total_dim, source_mode_dim))
                        if null_indices is not None:
                            x_indices, y_indices = null_indices
                            if source_idx in x_indices:
                                additional_indices = list(np.asarray(
                                    y_indices)[np.asarray(x_indices) == source_idx])
                                remove_idx.extend(additional_indices)
                        log_likelihood_gradient = np.delete(
                            log_likelihood_gradient, remove_idx)
                        log_likelihood_hessian = np.delete(
                            log_likelihood_hessian, remove_idx, axis=0)
                        log_likelihood_hessian = np.delete(
                            log_likelihood_hessian, remove_idx, axis=1)

        bias_total += log_likelihood_gradient.dot(np.linalg.solve(
            log_likelihood_hessian, log_likelihood_gradient))

    return bias_total


def compute_bias_by_index(source_index, covariance_matrix, coefficient_matrix, data_matrix,
                          s_bar, b, m, p, null_indices=None):
    """
    Computes the bias in the deviance for a specific source index.

    Parameters:
        source_index (int): The index of the source.
        covariance_matrix (ndarray): Covariance matrix of shape (n_sources*mo, n_sources*mo).
        coefficient_matrix (ndarray): Coefficient matrix of shape (n_sources*mo, n_sources*order*mo).
        data_matrix (ndarray): Data matrix of shape (t, n_sources*mo).
        s_bar (float): A parameter.
        b (float): A parameter.
        m (int): A parameter.
        p (int): A parameter.
        null_indices (tuple or None): Indices where cross-talk components are forced to zero.

    Returns:
        float: Computed bias value.
    """
    warnings.filterwarnings('always')
    _, total_coeff_dim = coefficient_matrix.shape

    # Calculate s1, s2, s3, and n using calculate_ss function
    # We don't need s3 in this function
    s1, s2, _, n_samples = calculate_ss(data_matrix, s_bar, b, m, p)

    source_coefficients = coefficient_matrix[source_index]
    source_covariance = covariance_matrix[source_index, source_index]

    # Compute the gradient of log-likelihood and the Hessian of log-likelihood
    gradient_log_likelihood = (-s2.dot(source_coefficients) +
                               s1[source_index]) / source_covariance
    hessian_log_likelihood = -s2 / source_covariance

    if null_indices is not None:
        x_index, y_index = null_indices
        if source_index in x_index:
            removed_indices = np.asarray(
                y_index)[np.asarray(x_index) == source_index]
            gradient_log_likelihood = np.delete(
                gradient_log_likelihood, removed_indices)
            hessian_log_likelihood = np.delete(
                hessian_log_likelihood, removed_indices, axis=0)
            hessian_log_likelihood = np.delete(
                hessian_log_likelihood, removed_indices, axis=1)

    try:
        cholesky_factor = linalg.cho_factor(-hessian_log_likelihood)
        temp_solve = linalg.cho_solve(cholesky_factor, gradient_log_likelihood)
        computed_bias = n_samples * gradient_log_likelihood.dot(temp_solve)
    except linalg.LinAlgError:
        warnings.warn(f'source-index {source_index} hessian_log_likelihood is not negative definite: '
                      'setting positive eigenvalues equal to zero, '
                      'result may not be accurate.', RuntimeWarning, stacklevel=2)
        eigenvalues, eigenvectors = linalg.eigh(-hessian_log_likelihood)
        positive_eigenvalues_indices = eigenvalues > 0
        temp_solve = eigenvectors.dot(gradient_log_likelihood)
        computed_bias = np.sum(
            temp_solve[positive_eigenvalues_indices] ** 2 / eigenvalues[positive_eigenvalues_indices])
        computed_bias *= n_samples

    return computed_bias
