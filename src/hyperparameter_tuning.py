import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from .models import ReturnBasedRegressor


def tune_ridge(X_train: pd.DataFrame, y_train: pd.Series, close_train: pd.Series, cv_splits: int = 5) -> Tuple[ReturnBasedRegressor, Dict[str, Any]]:
    y_returns = (y_train - close_train) / close_train
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("regressor", Ridge(random_state=42))
    ])
    param_grid = {"regressor__alpha": [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]}
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_returns)
    return ReturnBasedRegressor(grid.best_estimator_), grid.best_params_


def tune_lasso(X_train: pd.DataFrame, y_train: pd.Series, close_train: pd.Series, cv_splits: int = 5) -> Tuple[ReturnBasedRegressor, Dict[str, Any]]:
    y_returns = (y_train - close_train) / close_train
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("regressor", Lasso(random_state=42, max_iter=10000, tol=1e-3))
    ])
    param_grid = {"regressor__alpha": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3]}
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_returns)
    return ReturnBasedRegressor(grid.best_estimator_), grid.best_params_


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, close_train: pd.Series, n_iter: int = 10, cv_splits: int = 5) -> Tuple[ReturnBasedRegressor, Dict[str, Any]]:
    y_returns = (y_train - close_train) / close_train
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [100, 150, 200],
        "max_depth": [6, 8, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [2, 4],
        "max_features": ["sqrt", 0.8]
    }
    search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter, cv=tscv, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1)
    search.fit(X_train, y_returns)
    return ReturnBasedRegressor(search.best_estimator_), search.best_params_


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, close_train: pd.Series, n_iter: int = 10, cv_splits: int = 5) -> Tuple[ReturnBasedRegressor, Dict[str, Any]]:
    y_returns = (y_train - close_train) / close_train
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0]
    }
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=tscv, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1)
    search.fit(X_train, y_returns)
    return ReturnBasedRegressor(search.best_estimator_), search.best_params_
