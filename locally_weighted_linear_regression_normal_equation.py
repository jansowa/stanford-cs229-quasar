import numpy as np
from numpy.typing import ArrayLike
from typing import Callable


class LocallyWeightedLinearRegressionNormalEquation:
    X: np.ndarray
    y: np.ndarray
    weights_expression: Callable


    @staticmethod
    def _calculate_beta(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        X_ones = np.insert(X, 0, 1, axis=1)
        W = np.diag(weights)
        t_dot = X_ones.T.dot(W)
        dot = t_dot.dot(X_ones)
        return np.linalg.inv(dot) \
            .dot(X_ones.T).dot(W).dot(y)

    def fit(self, X: np.ndarray, y: np.ndarray, weights_expression: Callable) -> None:
        X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
        self.X = X_temp
        self.y = y
        self.weights_expression = weights_expression

    def predict(self, X: np.ndarray) -> ArrayLike:
        X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
        predictions = []
        for sample in X_temp:
            weights = self.weights_expression(sample, self.X).reshape(-1)
            beta = LocallyWeightedLinearRegressionNormalEquation._calculate_beta(self.X, self.y, weights)
            prediction = LocallyWeightedLinearRegressionNormalEquation._calculate_targets(sample, beta)
            predictions += [prediction]
        return predictions

    @staticmethod
    def _calculate_targets(X: np.ndarray, beta: np.ndarray) -> ArrayLike:
        X_temp = np.array([X.tolist()])
        X_ones = np.insert(X_temp, 0, 1, axis=1)
        return X_ones.dot(beta)[0]
