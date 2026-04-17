"""
Model definitions for the Walmart demand forecasting project.

Provides a unified interface for training and generating predictions with:
  - LinearRegressionModel  (baseline)
  - RandomForestModel
  - XGBoostModel
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from walmart_forecasting.feature_engineering import ALL_FEATURES, TARGET


class _BaseModel:
    """Shared fit / predict / save / load interface."""

    name: str = "base"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_BaseModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        print(f"[{self.name}] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "_BaseModel":
        return joblib.load(path)

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Return the subset of ALL_FEATURES that exist in *df*."""
        return [c for c in ALL_FEATURES if c in df.columns]


class LinearRegressionModel(_BaseModel):
    """Baseline linear model with standard scaling."""

    name = "linear_regression"

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self._feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionModel":
        self._feature_cols = self._get_feature_cols(X)
        X_arr = self.scaler.fit_transform(X[self._feature_cols].fillna(0))
        self.model.fit(X_arr, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = self.scaler.transform(X[self._feature_cols].fillna(0))
        return self.model.predict(X_arr)


class RandomForestModel(_BaseModel):
    """Random Forest regressor – good balance of accuracy and interpretability."""

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self._feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self._feature_cols = self._get_feature_cols(X)
        self.model.fit(X[self._feature_cols].fillna(0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self._feature_cols].fillna(0))

    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self._feature_cols,
        ).sort_values(ascending=False)


class XGBoostModel(_BaseModel):
    """Gradient-boosted trees – typically highest accuracy on tabular data."""

    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
        )
        self._feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self._feature_cols = self._get_feature_cols(X)
        self.model.fit(X[self._feature_cols].fillna(0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self._feature_cols].fillna(0))

    def feature_importances(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self._feature_cols,
        ).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type] = {
    "linear_regression": LinearRegressionModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
}


def get_model(name: str, **kwargs) -> _BaseModel:
    """Instantiate a model by name.  Extra kwargs are forwarded to the constructor."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'.  Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
