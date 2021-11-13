import numpy as np


class LogisticRegression:
    r"""
    A simple logistic regression model fit via gradient descent on the
    penalized negative log likelihood.
    Notes
    -----
    For logistic regression, the penalized negative log likelihood of the
    targets **y** under the current model is
    .. math::
        - \log \mathcal{L}(\mathbf{b}, \mathbf{y}) = -\frac{1}{N} \left[
            \left(
                \sum_{i=0}^N y_i \log(\hat{y}_i) +
                  (1-y_i) \log(1-\hat{y}_i)
            \right) - R(\mathbf{b}, \gamma)
        \right]
    where
    .. math::
        R(\mathbf{b}, \gamma) = \left\{
            \begin{array}{lr}
                \frac{\gamma}{2} ||\mathbf{beta}||_2^2 & :\texttt{ penalty = 'l2'}\\
                \gamma ||\beta||_1 & :\texttt{ penalty = 'l1'}
            \end{array}
            \right.
    is a regularization penalty, :math:`\gamma` is a regularization weight,
    `N` is the number of examples in **y**, and **b** is the vector of model
    coefficients.
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

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self):
        pass
