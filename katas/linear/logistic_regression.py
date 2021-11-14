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
        assert penalty in [
            "l1",
            "l2",
        ], f"penalty can be either 'l1' or 'l2', but got: {penalty}"

        self.params = np.array([])

        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept

    def fit(
        self,
        samples: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.01,
        tolerance: float = 1e-7,
        max_iterations: int = 1e7,
    ) -> None:
        n_samples, n_features = samples.shape

        self.params = np.random.rand(n_features)

        previous_loss = np.inf

        for _ in range(max_iterations):
            target_predictions = self.predict(samples)
            # get loss
            loss = self._calculate_loss(targets, target_predictions, samples)

            if previous_loss - loss < tolerance:
                return

            # update params
            self.params -= learning_rate * self._get_gradient(targets, target_predictions, samples)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """
        Generates prediction probabilities based on new samples and the trained model

        Parameters
        ----------
        samples: :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` new examples, each of dimension `M`.

        Returns
        -------
        target_predictions: :py:class:`ndarray <numpy.ndarray>` of shape `(N, )`
            The model prediction probabilities for N samples
        """
        return self._get_sigmoid(samples @ self.params)

    def _calculate_loss(self, targets: np.ndarray, predictions: np.ndarray, samples: np.ndarray) -> float:
        r"""
        Penalized negative log likelihood (NLL) of the targets under the current
        model.

        .. math::

            \text{NLL} = -\frac{1}{N} \left[
                \left(
                    \sum_{i=0}^N y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)
                \right) - R(\mathbf{b}, \gamma)
            \right]
        """
        n_samples, n_features = samples.shape
        gamma, penalty = self.gamma, self.penalty
        normalized_params = np.linalg.norm(self.params, ord=2 if penalty == 'l2' else 1)

        nll = -np.log(predictions[targets == 1]).sum() - np.log(1 - predictions[targets == 0]).sum()
        regularization = 0.5 * gamma * normalized_params ** 2 if penalty == 'l2' else gamma * normalized_params

        return (nll + regularization) / n_samples

    def _get_gradient(self, targets: np.ndarray, predictions: np.ndarray, samples: np.ndarray) -> np.ndarray:
        r"""
        Gradient of the penalized negative log likelihood wrt params

        .. math::

            \text{Gradient} = -\frac{1}{N} \left[
                \left(
                    \sum_{i=0}^N (y_i - \hat{y}_i) x_i
                \right) + R(\mathbf{b}, \gamma)
            \right]
        """
        n_samples, n_features = samples.shape
        gamma, penalty = self.gamma, self.penalty

        reg_grad = gamma * self.params if penalty == 'l2' else gamma * np.sign(self.params)

        return -((targets - predictions) @ samples + reg_grad) / n_samples

    def _get_sigmoid(self, samples: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-samples))
