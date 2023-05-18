import numpy as np
from numpy.typing import ArrayLike


class LinearRegressionNormalEquation:
    _beta = None

    @staticmethod
    def _calculate_beta(X: ArrayLike, y: ArrayLike) -> ArrayLike:
        X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
        X_ones = np.insert(X_temp, 0, 1, axis=1)
        return np.matmul(
            np.matmul(
                np.linalg.inv(
                    np.matmul(X_ones.T, X_ones)),
                X_ones.T),
            y)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        self._beta = self._calculate_beta(X, y)

    def predict(self, X: ArrayLike) -> ArrayLike:
        if self._beta is None:
            raise RuntimeError("The model is not fitted.")
        return LinearRegressionNormalEquation._calculate_targets(X, self._beta)

    @staticmethod
    def _calculate_targets(X: ArrayLike, beta: ArrayLike) -> ArrayLike:
        X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
        X_ones = np.insert(X_temp, 0, 1, axis=1)
        return np.matmul(X_ones, beta)
