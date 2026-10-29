import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_stock_data, validate_stock_data
from src.news_collector import collect_financial_news
from src.sentiment_analyzer import run_sentiment_analysis
from src.feature_engineering import (
    create_market_features,
    merge_sentiment_with_market_data,
    get_feature_columns,
)
from src.models import get_all_models, get_pca_models, train_and_evaluate_models, BlendedRegressor
from src.evaluation import (
    compute_regression_metrics,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
    plot_sentiment_trends,
    plot_sentiment_distribution,
    plot_sentiment_model_comparison,
)
from src.hyperparameter_tuning import tune_ridge, tune_lasso, tune_random_forest, tune_xgboost


def main():
    print("=" * 75)
    print("   STOCK MARKET ANALYSIS, ML REGRESSION & NLP FINBERT SENTIMENT PIPELINE")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(base_dir, "data", "stock_data.csv")
    news_data_path = os.path.join(base_dir, "data", "financial_news.csv")
    news_sentiment_path = os.path.join(base_dir, "data", "financial_news_sentiment.csv")
    daily_sentiment_path = os.path.join(base_dir, "data", "daily_sentiment.csv")
    processed_data_path = os.path.join(base_dir, "data", "stock_data_processed.csv")
    sentiment_data_path = os.path.join(base_dir, "data", "stock_data_with_sentiment.csv")
    figures_dir = os.path.join(base_dir, "outputs", "figures")
    metrics_path = os.path.join(base_dir, "outputs", "metrics_summary.csv")
    comparison_path = os.path.join(base_dir, "outputs", "sentiment_model_comparison.csv")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load & Validate Market Data
    print("\n[1/8] Loading and Validating Market Data...")
    df_raw = load_stock_data(raw_data_path)
    val = validate_stock_data(df_raw)
    print(f"Total Trading Days: {val['Total_Rows']}")
    print(f"Date Range: {val['Date_Range']}")
    print(f"Price Range: ${val['Min_Close']:.2f} to ${val['Max_Close']:.2f} (Mean: ${val['Mean_Close']:.2f})")

    # 2. Collect Financial News & Run FinBERT Sentiment Scoring
    print("\n[2/8] Collecting Financial News & Generating FinBERT Sentiment Scores...")
    news_df = collect_financial_news(save_path=news_data_path)
    classified_news_df, daily_sentiment_df = run_sentiment_analysis(
        news_df,
        classified_save_path=news_sentiment_path,
        daily_save_path=daily_sentiment_path
    )
    print(f"Scored {len(classified_news_df)} articles across {len(daily_sentiment_df)} trading dates.")

    # 3. Exploratory Data Analysis & Sentiment Visualizations
    print("\n[3/8] Generating EDA Visualizations & Sentiment Trends...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df_raw["Date"], df_raw["Close"], color="#1f77b4", linewidth=1.5, label="Close Price ($)")
    ax1.plot(df_raw["Date"], df_raw["Close"].rolling(50).mean(), color="#ff7f0e", linewidth=1.2, linestyle="--", label="50-Day SMA")
    ax1.plot(df_raw["Date"], df_raw["Close"].rolling(200).mean(), color="#2ca02c", linewidth=1.2, linestyle=":", label="200-Day SMA")
    ax1.set_title("Stock Price History & Moving Averages (2000 - 2026)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Price ($)", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.bar(df_raw["Date"], df_raw["Volume"] / 1e6, color="#7f7f7f", alpha=0.6, width=2)
    ax2.set_ylabel("Volume (M)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "eda_price_history.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    clean_returns = df_raw["Daily_Return"].dropna()
    sns.histplot(clean_returns, bins=60, kde=True, color="#3b528b", ax=ax1)
    ax1.axvline(0, color="red", linestyle="--", alpha=0.7)
    ax1.set_title(f"Daily Returns Distribution (Skew: {clean_returns.skew():.2f}, Kurtosis: {clean_returns.kurtosis():.2f})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Daily Return", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)

    rolling_vol = df_raw["Daily_Return"].rolling(30).std() * np.sqrt(252) * 100
    ax2.plot(df_raw["Date"], rolling_vol, color="#d62728", linewidth=1.2)
    ax2.set_title("Annualized 30-Day Rolling Volatility (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Volatility (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "eda_returns_and_volatility.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plot_sentiment_trends(
        daily_sentiment_df, df_raw,
        save_path=os.path.join(figures_dir, "sentiment_vs_stock_price.png")
    )
    plot_sentiment_distribution(
        classified_news_df,
        save_path=os.path.join(figures_dir, "sentiment_distribution.png")
    )

    # 4. Feature Engineering & Sentiment Merging
    print("\n[4/8] Engineering Features & Merging FinBERT Sentiment (Zero Leakage)...")
    df_featured = create_market_features(df_raw, drop_na=True)
    df_featured.to_csv(processed_data_path, index=False)
    
    df_with_sentiment = merge_sentiment_with_market_data(df_featured, daily_sentiment_df)
    df_with_sentiment.to_csv(sentiment_data_path, index=False)

    feature_cols_base = get_feature_columns(df_with_sentiment, include_sentiment=False)
    feature_cols_sent = get_feature_columns(df_with_sentiment, include_sentiment=True)
    print(f"Baseline features: {len(feature_cols_base)} technical indicators.")
    print(f"Sentiment-enhanced features: {len(feature_cols_sent)} (technical + FinBERT signals).")

    corr_subset = ["Close", "Target_Next_Close", "Daily_Return", "HL_Spread", "SMA_20", "SMA_50", "SMA_200", "RSI_14", "MACD", "BB_Width", "ATR_14", "Volume", "OBV", "Close_Lag_1"]
    if "Sentiment_Mean" in df_with_sentiment.columns:
        corr_subset.extend(["Sentiment_Mean", "News_Volume", "Sentiment_Lag_1"])
    plt.figure(figsize=(12, 10))
    corr_matrix = df_with_sentiment[corr_subset].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title("Correlation Matrix of Market Factors & FinBERT Sentiment", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "eda_correlation_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    pair_cols = ["Close", "Daily_Return", "Volume", "SMA_20", "RSI_14", "MACD", "Target_Next_Close"]
    sample_df = df_with_sentiment[pair_cols].iloc[::5]
    g = sns.pairplot(sample_df, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15, "color": "#1f77b4"}, diag_kws={"color": "#1f77b4"})
    g.fig.suptitle("Pair Plot of Key Market Factors & Target", y=1.02, fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(figures_dir, "eda_pair_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Chronological Train-Val-Test Split
    print("\n[5/8] Splitting Chronologically (70% Train, 15% Val, 15% Test)...")
    n_total = len(df_with_sentiment)
    train_idx = int(n_total * 0.70)
    val_idx = int(n_total * 0.85)

    train_df = df_with_sentiment.iloc[:train_idx]
    val_df = df_with_sentiment.iloc[train_idx:val_idx]
    test_df = df_with_sentiment.iloc[val_idx:]

    close_train = train_df["Close"]
    close_val = val_df["Close"]
    close_test = test_df["Close"]

    y_train = train_df["Target_Next_Close"]
    y_val = val_df["Target_Next_Close"]
    y_test = test_df["Target_Next_Close"]

    X_train_base, X_val_base, X_test_base = train_df[feature_cols_base], val_df[feature_cols_base], test_df[feature_cols_base]
    X_train_sent, X_val_sent, X_test_sent = train_df[feature_cols_sent], val_df[feature_cols_sent], test_df[feature_cols_sent]

    # 6. Model Training & Comparison: Baseline vs Sentiment-Enhanced
    print("\n[6/8] Training Models: Baseline (Technical Only) vs Sentiment-Enhanced...")

    # A. Baseline Models
    print("  -> Training Baseline Models...")
    models_base = get_all_models()
    metrics_base_df, trained_models_base, preds_base = train_and_evaluate_models(
        X_train_base, y_train, X_val_base, y_val, X_test_base, y_test,
        close_train, close_val, close_test, models=models_base
    )
    pca_base_models = get_pca_models(n_components=0.95)
    pca_metrics_base, trained_pca_base, preds_pca_base = train_and_evaluate_models(
        X_train_base, y_train, X_val_base, y_val, X_test_base, y_test,
        close_train, close_val, close_test, models=pca_base_models
    )
    trained_models_base.update(trained_pca_base)
    preds_base.update(preds_pca_base)

    best_ridge_b, _ = tune_ridge(X_train_base, y_train, close_train, cv_splits=5)
    best_lasso_b, _ = tune_lasso(X_train_base, y_train, close_train, cv_splits=5)
    best_rf_b, _ = tune_random_forest(X_train_base, y_train, close_train, n_iter=10, cv_splits=5)
    best_xgb_b, _ = tune_xgboost(X_train_base, y_train, close_train, n_iter=10, cv_splits=5)

    tuned_base = {
        "Tuned Ridge": best_ridge_b,
        "Tuned Lasso": best_lasso_b,
        "Tuned Random Forest": best_rf_b,
        "Tuned XGBoost": best_xgb_b
    }
    tuned_results_base = []
    for name, model in tuned_base.items():
        trained_models_base[name] = model
        ptr = model.predict(X_train_base, close_train)
        pv = model.predict(X_val_base, close_val)
        pte = model.predict(X_test_base, close_test)
        preds_base[name] = {"train": ptr, "val": pv, "test": pte}
        tuned_results_base.append(compute_regression_metrics(y_train, ptr, close_train, model_name=name, split_name="Train"))
        tuned_results_base.append(compute_regression_metrics(y_val, pv, close_val, model_name=name, split_name="Validation"))
        tuned_results_base.append(compute_regression_metrics(y_test, pte, close_test, model_name=name, split_name="Test"))

    val_rmses_b = [compute_regression_metrics(y_val, preds_base[m]["val"])["RMSE"] for m in tuned_base]
    weights_b = [1.0 / (r + 1e-6) for r in val_rmses_b]
    blended_base = BlendedRegressor(models=[(m, tuned_base[m]) for m in tuned_base], weights=weights_b)
    blended_base.fit(X_train_base, y_train, close_train)
    trained_models_base["Tuned Ensemble"] = blended_base
    ptr_b = blended_base.predict(X_train_base, close_train)
    pv_b = blended_base.predict(X_val_base, close_val)
    pte_b = blended_base.predict(X_test_base, close_test)
    preds_base["Tuned Ensemble"] = {"train": ptr_b, "val": pv_b, "test": pte_b}
    tuned_results_base.append(compute_regression_metrics(y_train, ptr_b, close_train, model_name="Tuned Ensemble", split_name="Train"))
    tuned_results_base.append(compute_regression_metrics(y_val, pv_b, close_val, model_name="Tuned Ensemble", split_name="Validation"))
    tuned_results_base.append(compute_regression_metrics(y_test, pte_b, close_test, model_name="Tuned Ensemble", split_name="Test"))

    all_base_df = pd.concat([metrics_base_df, pca_metrics_base, pd.DataFrame(tuned_results_base)], ignore_index=True)
    all_base_df["Feature_Set"] = "Baseline (Technical Only)"

    # B. Sentiment-Enhanced Models
    print("  -> Training Sentiment-Enhanced Models...")
    models_sent = get_all_models()
    metrics_sent_df, trained_models_sent, preds_sent = train_and_evaluate_models(
        X_train_sent, y_train, X_val_sent, y_val, X_test_sent, y_test,
        close_train, close_val, close_test, models=models_sent
    )
    pca_sent_models = get_pca_models(n_components=0.95)
    pca_metrics_sent, trained_pca_sent, preds_pca_sent = train_and_evaluate_models(
        X_train_sent, y_train, X_val_sent, y_val, X_test_sent, y_test,
        close_train, close_val, close_test, models=pca_sent_models
    )
    trained_models_sent.update(trained_pca_sent)
    preds_sent.update(preds_pca_sent)

    best_ridge_s, _ = tune_ridge(X_train_sent, y_train, close_train, cv_splits=5)
    best_lasso_s, _ = tune_lasso(X_train_sent, y_train, close_train, cv_splits=5)
    best_rf_s, _ = tune_random_forest(X_train_sent, y_train, close_train, n_iter=10, cv_splits=5)
    best_xgb_s, _ = tune_xgboost(X_train_sent, y_train, close_train, n_iter=10, cv_splits=5)

    tuned_sent = {
        "Tuned Ridge": best_ridge_s,
        "Tuned Lasso": best_lasso_s,
        "Tuned Random Forest": best_rf_s,
        "Tuned XGBoost": best_xgb_s
    }
    tuned_results_sent = []
    for name, model in tuned_sent.items():
        trained_models_sent[name] = model
        ptr = model.predict(X_train_sent, close_train)
        pv = model.predict(X_val_sent, close_val)
        pte = model.predict(X_test_sent, close_test)
        preds_sent[name] = {"train": ptr, "val": pv, "test": pte}
        tuned_results_sent.append(compute_regression_metrics(y_train, ptr, close_train, model_name=name, split_name="Train"))
        tuned_results_sent.append(compute_regression_metrics(y_val, pv, close_val, model_name=name, split_name="Validation"))
        tuned_results_sent.append(compute_regression_metrics(y_test, pte, close_test, model_name=name, split_name="Test"))

    val_rmses_s = [compute_regression_metrics(y_val, preds_sent[m]["val"])["RMSE"] for m in tuned_sent]
    weights_s = [1.0 / (r + 1e-6) for r in val_rmses_s]
    blended_sent = BlendedRegressor(models=[(m, tuned_sent[m]) for m in tuned_sent], weights=weights_s)
    blended_sent.fit(X_train_sent, y_train, close_train)
    trained_models_sent["Tuned Ensemble"] = blended_sent
    ptr_s = blended_sent.predict(X_train_sent, close_train)
    pv_s = blended_sent.predict(X_val_sent, close_val)
    pte_s = blended_sent.predict(X_test_sent, close_test)
    preds_sent["Tuned Ensemble"] = {"train": ptr_s, "val": pv_s, "test": pte_s}
    tuned_results_sent.append(compute_regression_metrics(y_train, ptr_s, close_train, model_name="Tuned Ensemble", split_name="Train"))
    tuned_results_sent.append(compute_regression_metrics(y_val, pv_s, close_val, model_name="Tuned Ensemble", split_name="Validation"))
    tuned_results_sent.append(compute_regression_metrics(y_test, pte_s, close_test, model_name="Tuned Ensemble", split_name="Test"))

    all_sent_df = pd.concat([metrics_sent_df, pca_metrics_sent, pd.DataFrame(tuned_results_sent)], ignore_index=True)
    all_sent_df["Feature_Set"] = "Sentiment-Enhanced"

    # Save combined metrics
    full_metrics_df = pd.concat([all_base_df, all_sent_df], ignore_index=True)
    full_metrics_df.to_csv(metrics_path, index=False)

    # 7. Comparison & Visualizations
    print("\n[7/8] Generating Benchmark Visualizations & Model Comparisons...")

    # PCA explained variance plot (from fitted object)
    fitted_pca = trained_pca_sent["PCA + Ridge"].base_estimator.named_steps["pca"]
    var_ratio = fitted_pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    comps = np.arange(1, len(var_ratio) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(comps, var_ratio * 100, alpha=0.6, color="#1f77b4", label="Individual Explained Variance (%)")
    ax1.plot(comps, cum_var * 100, color="#d62728", marker="o", linewidth=1.5, label="Cumulative Explained Variance (%)")
    ax1.axhline(95, color="black", linestyle="--", alpha=0.7, label="95% Threshold")
    ax1.set_xlabel("Principal Component Index", fontsize=11)
    ax1.set_ylabel("Explained Variance (%)", fontsize=11)
    ax1.set_title(f"PCA Explained Variance ({len(var_ratio)} Components retained for >=95% variance)", fontsize=13, fontweight="bold")
    ax1.legend(loc="center right")
    ax1.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "pca_explained_variance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Model Comparison Leaderboard (Test Set)
    test_comp = full_metrics_df[full_metrics_df["Split"] == "Test"].sort_values("RMSE").reset_index(drop=True)
    test_comp.to_csv(comparison_path, index=False)

    # Comparative Plot
    plot_sentiment_model_comparison(test_comp, save_path=os.path.join(figures_dir, "sentiment_model_comparison.png"))

    # Single Leaderboard Bar Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=test_comp[test_comp["Feature_Set"] == "Sentiment-Enhanced"], x="RMSE", y="Model", hue="Model", palette="viridis", legend=False)
    plt.title("Sentiment-Enhanced Model Comparison on Test Set (RMSE)", fontsize=13, fontweight="bold")
    plt.xlabel("Test RMSE ($) - Lower is Better", fontsize=11)
    plt.ylabel("Model", fontsize=11)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_comparison_rmse.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Best model predictions and residuals
    best_model_name = test_comp.iloc[0]["Model"]
    best_feature_set = test_comp.iloc[0]["Feature_Set"]
    if best_feature_set == "Sentiment-Enhanced":
        best_pred = preds_sent[best_model_name]["test"]
    else:
        best_pred = preds_base[best_model_name]["test"]

    plot_actual_vs_predicted(
        test_df["Date"], y_test.values, best_pred,
        model_name=f"{best_model_name} ({best_feature_set})",
        title=f"Test Set: Actual vs Predicted Price ({best_model_name})",
        save_path=os.path.join(figures_dir, "best_model_predictions_test.png")
    )
    plot_residuals(
        y_test.values, best_pred,
        model_name=f"{best_model_name} ({best_feature_set})",
        save_path=os.path.join(figures_dir, "best_model_residuals.png")
    )
    plot_feature_importance(
        feature_cols_sent, best_rf_s.base_estimator.feature_importances_,
        top_n=15, title="Random Forest Feature Importance (Sentiment-Enhanced)",
        save_path=os.path.join(figures_dir, "feature_importance_rf.png")
    )
    plot_feature_importance(
        feature_cols_sent, best_xgb_s.base_estimator.feature_importances_,
        top_n=15, title="XGBoost Feature Importance (Sentiment-Enhanced)",
        save_path=os.path.join(figures_dir, "feature_importance_xgb.png")
    )

    # 8. Results Summary
    print("\n" + "=" * 75)
    print("                     TEST SET BENCHMARK RESULTS")
    print("=" * 75)
    summary_cols = ["Model", "Feature_Set", "RMSE", "MAE", "MAPE (%)", "R2 Score", "Directional Accuracy (%)"]
    print(test_comp[summary_cols].to_string(index=False))
    print("=" * 75)
    print(f"\n[OK] Pipeline completed successfully. Output files saved in 'outputs/' and 'data/'.")


if __name__ == "__main__":
    main()
