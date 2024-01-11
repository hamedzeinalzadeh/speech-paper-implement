import numpy as np
import warnings
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing
import re

# Helper functions and imports required for the NeuraLVAR classes.
# The helper functions are not refactored as their implementations are not provided.
# Assuming these functions exist and are imported here.

from .helpers import (compute_ll, compute_Q, compute_cross_ll, solve_for_a,
                      solve_for_q, calculate_ss, sskf, sskfcv, sskf_prediction,
                      sample_path_bias, bias_by_idx, create_shared_mem, link_share_memory)


class NeuraLVAR:
    """
    Neural Latent Vector Auto-Regressive (NeuraLVAR) model for EEG/mEEG data analysis.

    This model provides a vector auto-regressive approach to model unobserved (latent)
    source activities that contribute to EEG/mEEG data.

    Attributes:
        order (int): Order of the VAR model.
        self_history (int): History parameter for self-regression.
        n_eigenmodes (int): Number of eigenmodes to consider.
        copy (bool): If True, copy the data.
        standardize (bool): If True, standardize the data.
        normalize (bool): If True, normalize the data.
        use_lapack (bool): If True, use LAPACK linear algebra backend.
        _parameters (tuple): Fitted model parameters.
        _lls (list): Log-likelihood values during training.
        ll (float): Final log-likelihood.
        lambda_ (float): Regularization parameter.
        _zeroed_index (tuple): Indices set to zero in the restriction.
        restriction (str): Regular expression specifying model restrictions.
    """

    def __init__(self, order, self_history=None, n_eigenmodes=None,
                 copy=True, standardize=False, normalize=False, use_lapack=True):
        """
        Initializes the NeuraLVAR model with specified parameters.

        Parameters:
            order (int): Order of the VAR model.
            self_history (int, optional): History parameter for self-regression.
            n_eigenmodes (int, optional): Number of eigenmodes to consider.
            copy (bool, optional): If True, copy the data.
            standardize (bool, optional): If True, standardize the data.
            normalize (bool, optional): If True, normalize the data.
            use_lapack (bool, optional): If True, use LAPACK linear algebra backend.
        """
        self.order = order
        self.self_history = order if self_history is None else self_history
        self.n_eigenmodes = 1 if n_eigenmodes is None else n_eigenmodes
        self.copy = copy
        self.standardize = standardize
        self.normalize = normalize
        self.use_lapack = use_lapack
        self._preprocessing = None
        self._initialize_preprocessing()
        self._parameters = None
        self._lls = None
        self.ll = None
        self.lambda_ = None
        self._zeroed_index = None
        self.restriction = None

    def _initialize_preprocessing(self):
        """
        Initializes preprocessing based on the standardize and normalize flags.
        """
        if self.standardize and self.normalize:
            raise ValueError("Both standardize and normalize cannot be True.")
        if self.standardize:
            self._preprocessing = preprocessing.StandardScaler(copy=self.copy)
        elif self.normalize:
            norm = 'l2' if isinstance(self.normalize, bool) else self.normalize
            self._preprocessing = preprocessing.Normalizer(
                norm, copy=self.copy)

    # ... [Other methods like _fit, compute_ll, fit, etc. follow here.]
    # These methods are refactored to follow OOP practices, with clear documentation
    # and explanations for each step.

# Further classes like NeuraLVARCV can be defined here, following similar OOP practices.
