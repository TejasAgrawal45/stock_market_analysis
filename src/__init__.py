"""
Stock Market Analysis & Machine Learning Package
"""

from .data_loader import load_stock_data, validate_stock_data
from .feature_engineering import create_market_features, merge_sentiment_with_market_data, get_feature_columns
from .models import get_all_models, get_pca_models, train_and_evaluate_models
from .evaluation import (
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
    plot_sentiment_trends,
    plot_sentiment_distribution,
    plot_sentiment_model_comparison,
)
from .news_collector import collect_financial_news
from .sentiment_analyzer import FinBERTSentimentAnalyzer, run_sentiment_analysis

__all__ = [
    "load_stock_data",
    "validate_stock_data",
    "create_market_features",
    "merge_sentiment_with_market_data",
    "get_feature_columns",
    "get_all_models",
    "get_pca_models",
    "train_and_evaluate_models",
    "compute_regression_metrics",
    "plot_actual_vs_predicted",
    "plot_residuals",
    "plot_feature_importance",
    "plot_sentiment_trends",
    "plot_sentiment_distribution",
    "plot_sentiment_model_comparison",
    "collect_financial_news",
    "FinBERTSentimentAnalyzer",
    "run_sentiment_analysis",
]
