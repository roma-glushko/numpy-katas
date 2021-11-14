import numpy as np


class LinearRegression:
    r"""
    Implementation of linear regression with SGD optimization
    """

    def __init__(self):
        self.params = np.array([])

    def fit(
        self,
        samples: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.01,
        tolerance: float = 1e-7,
        epochs: int = 1e7,
    ) -> np.ndarray:
        samples = self.augment_intercept(samples)
        targets = targets[:, np.newaxis]

        num_features: int = samples.shape[1]
        self.params = np.ones((num_features, 1))

        training_history: np.ndarray = np.zeros((epochs, 1))

        previous_loss = np.inf

        for epoch_idx in range(epochs):
            loss = self.get_cost(
                samples @ self.params,
                targets,
            )

            training_history[epoch_idx] = loss

            if previous_loss - loss < tolerance:
                return training_history

            self.params -= learning_rate * self.get_gradient(
                samples, targets
            )

        return training_history

    def predict(self, samples: np.ndarray) -> np.ndarray:
        samples = self.augment_intercept(samples)

        return samples @ self.params

    def augment_intercept(self, samples: np.ndarray):
        num_features: int = samples.shape[0]

        return np.hstack(
            (
                np.ones((num_features, 1)),
                samples,
            )
        )

    def get_cost(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        return (1 / (2 * targets.size)) * np.sum((predictions - targets) ** 2)

    def get_gradient(self, samples: np.ndarray, targets: np.ndarray) -> float:
        num_samples, _ = samples.shape

        return (1 / num_samples) * samples.T @ ((samples @ self.params) - targets)

    def score(self, samples: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate R2 coefficient
        """
        predictions = self.predict(samples)

        residual_square_sum: float = ((predictions - targets) ** 2).sum()
        total_error_square_sum: float = ((targets - targets.mean()) ** 2).sum()

        return 1 - (residual_square_sum / total_error_square_sum)
