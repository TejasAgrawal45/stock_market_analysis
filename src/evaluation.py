import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    median_absolute_error,
)
from typing import Dict, Any, List, Optional


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_current: np.ndarray) -> float:
    actual_direction = np.sign(y_true - y_current)
    predicted_direction = np.sign(y_pred - y_current)
    correct = (actual_direction == predicted_direction)
    return float(np.mean(correct) * 100)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_current: Optional[np.ndarray] = None,
    model_name: str = "Model",
    split_name: str = "Test"
) -> Dict[str, Any]:
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    
    metrics = {
        "Model": model_name,
        "Split": split_name,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "MAPE (%)": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "R2 Score": float(r2_score(y_true, y_pred)),
    }
    
    if y_current is not None:
        y_current = np.array(y_current).ravel()
        metrics["Directional Accuracy (%)"] = float(compute_directional_accuracy(y_true, y_pred, y_current))
        
    return metrics


def plot_actual_vs_predicted(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_true, label="Actual Price", color="#1f77b4", linewidth=1.5, alpha=0.9)
    plt.plot(dates, y_pred, label=f"Predicted ({model_name})", color="#ff7f0e", linewidth=1.2, linestyle="--", alpha=0.9)
    plt.title(title or f"Actual vs Predicted Stock Price - {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Price ($)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(y_pred, residuals, alpha=0.5, color="#2ca02c", s=25)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax1.set_title(f"Residuals vs Predicted - {model_name}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Price ($)", fontsize=11)
    ax1.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    sns.histplot(residuals, kde=True, ax=ax2, color="#1f77b4", bins=40)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax2.set_title(f"Residual Distribution - {model_name}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Residual Value", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    title: str = "Top Feature Importances",
    save_path: Optional[str] = None
):
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]
    
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_scores, align="center", color="#3b528b", alpha=0.85)
    plt.yticks(y_pos, top_features, fontsize=10)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Score", fontsize=11)
    plt.title(f"{title} (Top {top_n})", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sentiment_trends(
    sentiment_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    df_m = stock_df.copy()
    df_s = sentiment_df.copy()
    df_m["Date"] = pd.to_datetime(df_m["Date"])
    df_s["Date"] = pd.to_datetime(df_s["Date"])
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]})
    
    ax1.plot(df_m["Date"], df_m["Close"], color="#1f77b4", linewidth=1.5, label="Close Price ($)")
    if "SMA_50" in df_m.columns:
        ax1.plot(df_m["Date"], df_m["SMA_50"], color="#ff7f0e", linewidth=1.2, linestyle="--", label="50-Day SMA")
    ax1.set_title("Stock Price & FinBERT Sentiment Dynamics", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price ($)", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    colors = ["#2ca02c" if s >= 0 else "#d62728" for s in df_s["Sentiment_Mean"]]
    ax2.bar(df_s["Date"], df_s["Sentiment_Mean"], color=colors, alpha=0.6, width=2, label="Daily Sentiment Score")
    if "Sentiment_EMA_3" in df_s.columns:
        ax2.plot(df_s["Date"], df_s["Sentiment_EMA_3"], color="#1f77b4", linewidth=1.5, label="Sentiment 3-Day EMA")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Sentiment Score (-1 to +1)", fontsize=11)
    ax2.set_ylim(-1.05, 1.05)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    ax3.bar(df_s["Date"], df_s["News_Volume"], color="#7f7f7f", alpha=0.7, width=2)
    ax3.set_ylabel("News Count", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sentiment_distribution(
    classified_news_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if not classified_news_df.empty and "Sentiment_Label" in classified_news_df.columns:
        label_counts = classified_news_df["Sentiment_Label"].value_counts()
        colors = {"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"}
        bar_colors = [colors.get(l, "#1f77b4") for l in label_counts.index]
        sns.barplot(x=label_counts.index, y=label_counts.values, palette=bar_colors, ax=ax1)
        for i, val in enumerate(label_counts.values):
            pct = val / len(classified_news_df) * 100
            ax1.text(i, val + max(label_counts.values)*0.02, f"{val} ({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")
        ax1.set_title("FinBERT Sentiment Classification Breakdown", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Article Count", fontsize=11)
        ax1.set_xlabel("Sentiment Category", fontsize=11)
        ax1.grid(axis="y", linestyle=":", alpha=0.6)
        
        scores = classified_news_df["Sentiment_Score"].dropna()
        sns.histplot(scores, kde=True, bins=30, color="#3b528b", ax=ax2)
        ax2.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax2.axvline(scores.mean(), color="red", linestyle=":", linewidth=1.5, label=f"Mean: {scores.mean():.2f}")
        ax2.set_title("FinBERT Polarity Distribution (Positive - Negative)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Continuous Sentiment Score", fontsize=11)
        ax2.set_ylabel("Article Frequency", fontsize=11)
        ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sentiment_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.barplot(
        data=comparison_df,
        x="RMSE", y="Model",
        hue="Feature_Set",
        palette={"Baseline (Technical Only)": "#1f77b4", "Sentiment-Enhanced": "#2ca02c"},
        ax=ax1
    )
    ax1.set_title("Test Set RMSE: Baseline vs Sentiment-Enhanced (Lower is Better)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Test RMSE ($)", fontsize=11)
    ax1.set_ylabel("Model", fontsize=11)
    ax1.grid(axis="x", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    sns.barplot(
        data=comparison_df,
        x="Directional Accuracy (%)", y="Model",
        hue="Feature_Set",
        palette={"Baseline (Technical Only)": "#1f77b4", "Sentiment-Enhanced": "#2ca02c"},
        ax=ax2
    )
    ax2.set_title("Test Directional Accuracy: Baseline vs Sentiment-Enhanced (Higher is Better)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Directional Accuracy (%)", fontsize=11)
    ax2.set_ylabel("", fontsize=11)
    ax2.grid(axis="x", linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

