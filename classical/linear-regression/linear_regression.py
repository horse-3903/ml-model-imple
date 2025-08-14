from __future__ import annotations

import os
from os import PathLike

import numpy as np
from numpy.typing import NDArray
from numpy import float64

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

target_col = "z"
n_features = 2

df1 = pd.read_csv("data/linear.csv")

X1 = df1.drop(columns=[target_col]).to_numpy()
y1 = df1[target_col].to_numpy()

df2 = pd.read_csv("data/planar.csv")
n_features = len(df2.columns) - 1

X2 = df2.drop(columns=[target_col]).to_numpy()
y2 = df2[target_col].to_numpy()

class LinearRegressionModel:
    def __init__(
            self, 
            w: NDArray[float64] = np.zeros(n_features), 
            b: float = 0.0,
            X_mean: float = 0.0,
            X_std: float = 0.0,
            y_mean: float = 0.0,
            y_std: float = 0.0,
        ) -> None:

        self.w = w
        self.b = b

        self.X_mean = X_mean
        self.X_std = X_std

        self.y_mean = y_mean
        self.y_std = y_std

        self.has_train = False
        self.cost_history = []
    
    # === Private Functions ===
    
    def _cost_function(self, X: NDArray[float64], y: NDArray[float64], w: NDArray[float64], b: float):
        m = y.shape[0]

        # C = 1/2m * sum(w_i * X_i + b)
        y_hat = self._predict_linear(X)
        
        r = y_hat - y
        r_squared = r @ r

        return r_squared / (2 * m)

    def _normalise_x(self, X: NDArray[float64], inplace: bool = False):
        if inplace:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0).clip(min=1e-8)
        
        X = (X - self.X_mean) / self.X_std
        return X

    def _normalise_y(self, y: NDArray[float64], inplace: bool = False):
        if inplace:
            self.y_mean = y.mean()
            self.y_std = y.std()
        
        y = (y - self.y_mean) / self.y_std
        return y

    def _denormalise_y(self, y: NDArray[float64]) -> NDArray[float64]:
        y = y * self.y_std + self.y_mean
        return y

    def _predict_linear(self, X: NDArray[float64]):
        return X @ self.w + self.b
    
    def _gradient_func(self, X: NDArray[float64], y: NDArray[float64]):
        m = y.shape[0]
        r = self._predict_linear(X) - y
        
        # d_C / d_w = 1/m * sum(X_i * (w_i * X_i + b - y_i))
        # d_C / d_w = 1/m * sum(X_i * C)
        d_w = (X.T @ r) / m

        # d_C / d_b = 1/m * sum((w_i * X_i + b - y_i))
        # d_C / d_b = 1/m * sum(C)
        d_b = np.sum(r) / m

        return d_w, d_b
    
    # === Public Functions ===
    
    def train(
            self, 
            X: NDArray[float64], 
            y: NDArray[float64], 
            alpha: float = 1e-3, 
            epochs: int = 50000, 
            verbose: int = 500, 
            grad_desc_mode: str = "batch", 
            batch_size: int = 16
        ):
        X = self._normalise_x(X, inplace=True)
        y = self._normalise_y(y, inplace=True)

        for epoch in range(1, epochs+1):
            cost = self._cost_function(X, y, self.w, self.b)
            self.cost_history.append(cost)

            if verbose and epoch % verbose == 0:
                print(f"=== Epoch {epoch} / {epochs} === \nEpoch Loss : {cost} \nBest Loss : {min(self.cost_history)}")

            if grad_desc_mode == "batch":
                d_w, d_b = self._gradient_func(X, y)

                self.w -= alpha * d_w
                self.b -= alpha * d_b
            
            elif grad_desc_mode == "minibatch":
                assert batch_size

                indices = np.random.permutation(X.shape[0])
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                for start in range(0, X.shape[0], batch_size):
                    end = start + batch_size
                    X_b = X_shuffled[start:end]
                    y_b = y_shuffled[start:end]

                    d_w, d_b = self._gradient_func(X_b, y_b)

                    self.w -= alpha * d_w
                    self.b -= alpha * d_b

            elif grad_desc_mode == "stochastic":
                for idx in range(0, X.shape[0]):
                    X_i = X[idx].reshape(1, -1)
                    y_i = np.array([y[idx]])

                    d_w, d_b = self._gradient_func(X_i, y_i)

                    self.w -= alpha * d_w
                    self.b -= alpha * d_b
        
        self.has_train = True

    def predict(self, X: NDArray[float64]):
        assert self.has_train

        X = self._normalise_x(X)
        y = self._predict_linear(X)
        return y
    
    def score(self, X: NDArray[float64], y: NDArray[float64]):
        y_pred = self.predict(X)
        y_pred = self._denormalise_y(y_pred)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
    
    # == Saving and Loading ==
    
    def save(self, filename: str | PathLike[str]):
        np.savez(
            filename,
            w=self.w,
            b=self.b,
            X_mean=self.X_mean,
            X_std=self.X_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
            has_train=self.has_train,
            cost_history=np.array(self.cost_history)
        )

    @staticmethod
    def load(filename: str | PathLike[str]):
        data = np.load(filename)
        model = LinearRegressionModel(
            w=data["w"],
            b=data["b"],
            X_mean=data["X_mean"],
            X_std=data["X_std"],
            y_mean=data["y_mean"],
            y_std=data["y_std"],
        )

        model.has_train = True
        model.cost_history = data["cost_history"].tolist()
        return model
    
    # == Graphing ==

    def plot_cost(self, ax: Axes):
        ax.plot(model1.cost_history, label="Batch Gradient Descent")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cost")
        ax.set_title("Cost History Over Epochs")
        ax.legend()

    def plot_predictions_3d(self, ax: Axes3D, X: NDArray[float64], y: NDArray[float64], num_points: int=20):
        idx = np.random.choice(len(X), size=min(num_points, len(X)), replace=False)
        X_sample = X[idx]
        y_true = y[idx]

        # Sort by the first feature for a straight line
        sorted_idx = np.argsort(X_sample[:, 0])
        X_sample_sorted = X_sample[sorted_idx]
        y_true_sorted = y_true[sorted_idx]
        y_pred_sorted = self._predict_linear(self._normalise_x(X_sample_sorted))
        y_pred_sorted = self._denormalise_y(y_pred_sorted)

        # Assume first two features for 3D plot
        ax.scatter(X_sample_sorted[:, 0], X_sample_sorted[:, 1], y_true_sorted, label="Actual", marker='o', color='b') # type: ignore
        ax.plot(X_sample_sorted[:, 0], X_sample_sorted[:, 1], y_pred_sorted, label="Predicted", color='r')
        ax.set_title(f"3D Predictions vs Actual ({len(y_true_sorted)} samples)")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Target Value")
        ax.legend()

if __name__ == "__main__":
    pretrained_model = False
    model1_file = "classical/linear-regression/models/model1.npz"
    model2_file = "classical/linear-regression/models/model2.npz"

    if os.path.exists(model1_file) and pretrained_model:
        model1 = LinearRegressionModel.load(model1_file)
    else:
        model1 = LinearRegressionModel()
        model1.train(X1, y1, grad_desc_mode="batch", epochs=50000)
        model1.save(model1_file)

    if os.path.exists(model2_file) and pretrained_model:
        model2 = LinearRegressionModel.load(model2_file)
    else:
        model2 = LinearRegressionModel()
        model2.train(X2, y2, grad_desc_mode="batch", epochs=50000)
        model2.save(model2_file)

    score1 = model1.score(X1, y1)
    score2 = model2.score(X2, y2)

    print(score1, score2)

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(2, 2, 1)
    model1.plot_cost(ax1)

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    model1.plot_predictions_3d(ax2, X1, y1, num_points=2000)

    ax3 = fig.add_subplot(2, 2, 3)
    model2.plot_cost(ax3)

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    model2.plot_predictions_3d(ax4, X2, y2, num_points=2000)

    plt.tight_layout()
    plt.show()