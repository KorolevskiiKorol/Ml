import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, Normalizer, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


class ElasticNetModelLab1:
    results = []

    @classmethod
    def show_best_r2(cls):

        df = pd.DataFrame(cls.results)

        df = df.sort_values(
        by="val_r2",
        ascending=False)
        return df

    def __init__(
        self,
        name,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaling=None,
        degree=1,
        alpha=0.05,
        l1_ratio=0.5,
        max_iter=10000
    ):
        self.name = name

        self.X_train = X_train
        self.y_train = y_train

        self.X_val = X_val
        self.y_val = y_val

        self.X_test = X_test
        self.y_test = y_test

        self.scaling = scaling
        self.degree = degree
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter

        self.model = self._build_pipeline()
        self.metrics = {}

    def _build_pipeline(self):
        steps = []

        if self.scaling == "standard":
            steps.append(("scaler", StandardScaler()))

        elif self.scaling == "normalizer":
            steps.append(("normalizer", Normalizer()))


        steps.append((
            "poly",
            PolynomialFeatures(degree=self.degree)
        ))

        steps.append((
            "elasticnet",
            ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter
            )
        ))

        return Pipeline(steps)

    def learn(self):
        self.model.fit(self.X_train, self.y_train)

        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)
        y_test_pred = self.model.predict(self.X_test)

        self.metrics = {
            "name": self.name,
            "train_mse": mean_squared_error(self.y_train, y_train_pred),
            "val_mse": mean_squared_error(self.y_val, y_val_pred),
            "test_mse": mean_squared_error(self.y_test, y_test_pred),

            "train_rmse": np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            "val_rmse": np.sqrt(mean_squared_error(self.y_val, y_val_pred)),
            "test_rmse": np.sqrt(mean_squared_error(self.y_test, y_test_pred)),

            "train_r2": r2_score(self.y_train, y_train_pred),
            "val_r2": r2_score(self.y_val, y_val_pred),
            "test_r2": r2_score(self.y_test, y_test_pred),
        }
        ElasticNetModelLab1.results.append(self.metrics)

        self.print_metrics()
        self.plot_all_predictions()

        return self.model

    def print_metrics(self):
        print(f"\n{self.name}")
        print(f"Scaling: {self.scaling}")
        print("-" * 50)
        print(
            f"Train RMSE: {self.metrics['train_rmse']:.4f} | "
            f"R2: {self.metrics['train_r2']:.8f}"
        )
        print(
            f"Val   RMSE: {self.metrics['val_rmse']:.4f} | "
            f"R2: {self.metrics['val_r2']:.8f}"
        )
        print(
            f"Test  RMSE: {self.metrics['test_rmse']:.4f} | "
            f"R2: {self.metrics['test_r2']:.8f}"
        )

    def plot_predictions(self, X, y, title):
        y_pred = self.model.predict(X)

        plt.scatter(y, y_pred, alpha=0.5)

        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))

        plt.plot([min_val, max_val], [min_val, max_val], "r--")

        plt.xlabel("True y")
        plt.ylabel("Predicted y")
        plt.title(title)
        plt.grid(True)

    def plot_all_predictions(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        self.plot_predictions(self.X_train, self.y_train, "Train")

        plt.subplot(1, 3, 2)
        self.plot_predictions(self.X_val, self.y_val, "Validation")

        plt.subplot(1, 3, 3)
        self.plot_predictions(self.X_test, self.y_test, "Test")

        plt.tight_layout()
        plt.show()