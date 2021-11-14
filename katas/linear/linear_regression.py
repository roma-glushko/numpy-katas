import numpy as np


class LinearRegression:
    r"""
    Implementation of linear regression with SGD optimization
    """

    def __init__(self):
        self.coefficients: np.ndarray = np.array([])

    def fit(
        self,
        samples: np.ndarray,
        targets: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 1500,
    ) -> np.ndarray:
        samples = self.augment_intercept(samples)
        targets = targets[:, np.newaxis]

        num_features: int = samples.shape[1]
        self.coefficients = np.ones((num_features, 1))

        training_history: np.ndarray = np.zeros((epochs, 1))

        for epoch_idx in range(epochs):
            self.coefficients -= learning_rate * self.get_cost_derivative(
                samples, targets
            )

            training_history[epoch_idx] = self.get_cost(
                samples @ self.coefficients,
                targets,
            )

        return training_history

    def predict(self, samples: np.ndarray) -> np.ndarray:
        samples = self.augment_intercept(samples)

        return samples @ self.coefficients

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

    def get_cost_derivative(self, samples: np.ndarray, targets: np.ndarray) -> float:
        num_samples: int = samples.shape[0]

        return (1 / num_samples) * samples.T @ ((samples @ self.coefficients) - targets)

    def score(self, samples: np.ndarray, targets: np.ndarray) -> float:
        """
        Calculate R2 coefficient
        """
        predictions: np.ndarray = self.predict(samples)

        residual_square_sum: float = ((predictions - targets) ** 2).sum()
        total_error_square_sum: float = ((targets - targets.mean()) ** 2).sum()

        return 1 - (residual_square_sum / total_error_square_sum)
