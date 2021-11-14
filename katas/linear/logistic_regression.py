import numpy as np


class LogisticRegression:
    r"""
    A simple logistic regression model fit via gradient descent on the
    penalized negative log likelihood.

    Parameters
    ----------
    penalty : {'l1', 'l2'}
        The type of regularization penalty to apply on the coefficients
        `beta`. Default is 'l2'.
    gamma : float
        The regularization weight. Larger values correspond to larger
        regularization penalties, and a value of 0 indicates no penalty.
        Default is 0.
    fit_intercept : bool
        Whether to fit an intercept term in addition to the coefficients in
        b. If True, the estimates for `beta` will have `M + 1` dimensions,
        where the first dimension corresponds to the intercept. Default is
        True.
    """

    def __init__(
        self, penalty: str = "l1", gamma: float = 0, fit_intercept: bool = True
    ):
        self.coefficients: np.ndarray = np.array([])

        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept

    def fit(self, samples: np.ndarray, targets: np.ndarray):
        pass

    def predict(self, samples: np.ndarray) -> np.ndarray:
        pass
