import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


class ReturnBasedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimator: Any):
        self.base_estimator = base_estimator

    def fit(self, X: pd.DataFrame, y: pd.Series, close_series: pd.Series):
        # Target is next-day return: (Close_{t+1} - Close_t) / Close_t
        returns = (y - close_series) / close_series
        self.base_estimator.fit(X, returns)
        return self

    def predict(self, X: pd.DataFrame, close_series: pd.Series) -> np.ndarray:
        pred_returns = self.base_estimator.predict(X)
        return close_series.values * (1.0 + pred_returns)


class BlendedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, models: List[Tuple[str, Any]], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights

    def fit(self, X: pd.DataFrame, y: pd.Series, close_series: pd.Series):
        for name, model in self.models:
            model.fit(X, y, close_series)
        return self

    def predict(self, X: pd.DataFrame, close_series: pd.Series) -> np.ndarray:
        preds = np.column_stack([model.predict(X, close_series) for name, model in self.models])
        if self.weights is None:
            return np.mean(preds, axis=1)
        norm_weights = np.array(self.weights) / np.sum(self.weights)
        return np.dot(preds, norm_weights)


def get_all_models() -> Dict[str, Any]:
    return {
        "Linear Regression": ReturnBasedRegressor(
            Pipeline([
                ("scaler", RobustScaler()),
                ("regressor", LinearRegression())
            ])
        ),
        "Ridge Regression": ReturnBasedRegressor(
            Pipeline([
                ("scaler", RobustScaler()),
                ("regressor", Ridge(alpha=100.0, random_state=42))
            ])
        ),
        "Lasso Regression": ReturnBasedRegressor(
            Pipeline([
                ("scaler", RobustScaler()),
                ("regressor", Lasso(alpha=0.0001, random_state=42, max_iter=2000))
            ])
        ),
        "Random Forest": ReturnBasedRegressor(
            RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        ),
        "XGBoost": ReturnBasedRegressor(
            xgb.XGBRegressor(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1
            )
        )
    }


def get_pca_models(n_components: float = 0.95) -> Dict[str, Any]:
    return {
        "PCA + Linear": ReturnBasedRegressor(
            Pipeline([
                ("scaler", RobustScaler()),
                ("pca", PCA(n_components=n_components, svd_solver="full")),
                ("regressor", LinearRegression())
            ])
        ),
        "PCA + Ridge": ReturnBasedRegressor(
            Pipeline([
                ("scaler", RobustScaler()),
                ("pca", PCA(n_components=n_components, svd_solver="full")),
                ("regressor", Ridge(alpha=100.0, random_state=42))
            ])
        ),
        "PCA + Lasso": ReturnBasedRegressor(
            Pipeline([
                ("scaler", RobustScaler()),
                ("pca", PCA(n_components=n_components, svd_solver="full")),
                ("regressor", Lasso(alpha=0.0001, random_state=42, max_iter=2000))
            ])
        )
    }


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    close_train: pd.Series,
    close_val: pd.Series,
    close_test: pd.Series,
    models: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[str, np.ndarray]]]:
    from .evaluation import compute_regression_metrics

    if models is None:
        models = get_all_models()

    results = []
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train, close_train)
        trained_models[name] = model

        pred_train = model.predict(X_train, close_train)
        pred_val = model.predict(X_val, close_val)
        pred_test = model.predict(X_test, close_test)

        predictions[name] = {
            "train": pred_train,
            "val": pred_val,
            "test": pred_test
        }

        results.append(compute_regression_metrics(y_train, pred_train, close_train, model_name=name, split_name="Train"))
        results.append(compute_regression_metrics(y_val, pred_val, close_val, model_name=name, split_name="Validation"))
        results.append(compute_regression_metrics(y_test, pred_test, close_test, model_name=name, split_name="Test"))

    return pd.DataFrame(results), trained_models, predictions
