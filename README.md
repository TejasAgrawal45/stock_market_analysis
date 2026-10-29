# 📈 Stock Market Analysis & ML Regression with FinBERT Sentiment

An end-to-end Quantitative Machine Learning, NLP Sentiment Analysis, and Exploratory Data Analysis (EDA) system comparing linear and regularized baselines, dimensionality reduction, decision tree ensembles, and blended stacking:
1. **Linear Regression** (OLS with RobustScaler)
2. **Ridge Regression** ($L_2$ regularized regression, default and tuned)
3. **Lasso Regression** ($L_1$ regularized regression, default and tuned)
4. **PCA Experiments** (PCA + Linear, PCA + Ridge, PCA + Lasso with zero data leakage)
5. **Random Forest Regressor** (Non-linear bagging ensemble)
6. **XGBoost Regressor** (Gradient boosting on decision trees)
7. **Tuned Blended Ensemble** (Inverse-RMSE weighted combination of tuned models)
8. **FinBERT NLP Sentiment Integration** (Financial news extraction, ProsusAI/finbert transformer scoring, zero-leakage sentiment lags, and baseline vs. sentiment-enhanced benchmarking)

---

## 📂 Project Structure

```
stock_market_analysis/
│
├── data/
│   ├── stock_data.csv                  # Raw price dataset (6,559 trading days, 2000-2026)
│   ├── stock_data_processed.csv        # Enriched dataset with 87 technical factors
│   ├── financial_news.csv              # Collected financial news articles (deduplicated)
│   ├── financial_news_sentiment.csv    # FinBERT-scored news items (Positive/Negative/Neutral)
│   ├── daily_sentiment.csv             # Aggregated daily sentiment features & rolling metrics
│   └── stock_data_with_sentiment.csv   # Unified dataset (6,359 trading days, 96 factors)
│
├── src/
│   ├── __init__.py                     # Package initialization and public exports
│   ├── data_loader.py                  # Date parsing & chronological train/val/test splitting
│   ├── feature_engineering.py          # Technical indicators, sentiment merging & leak prevention
│   ├── news_collector.py               # Financial news API ingestion (Finnhub, Alpha Vantage, RSS)
│   ├── sentiment_analyzer.py           # ProsusAI/finbert Transformer inference & daily aggregation
│   ├── models.py                       # Core regression models, PCA & blended ensemble
│   ├── evaluation.py                   # Regression & directional metrics + visualization suite
│   └── hyperparameter_tuning.py        # TimeSeriesSplit cross-validation tuning
│
├── outputs/
│   ├── figures/                        # Generated EDA, evaluation & sentiment charts
│   ├── metrics_summary.csv             # Full benchmark leaderboard across all models
│   └── sentiment_model_comparison.csv  # Direct baseline vs. sentiment-enhanced model comparison
│
├── stock_market_analysis.ipynb         # Interactive Jupyter Notebook
├── run_analysis.py                     # Standalone CLI execution pipeline (Steps 1–8)
├── requirements.txt                    # Python package requirements
├── .env.example                        # Secure API key template
└── README.md                           # Documentation and usage guide
```

---

## ⚙️ Setup & Configuration

### 1. Install Dependencies
Ensure you are using Python 3.10+ (recommended in a virtual environment):
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
The system supports API news collection from multiple sources:
- **Finnhub** (`FINNHUB_API_KEY`)
- **Alpha Vantage** (`ALPHAVANTAGE_API_KEY`)
- **NewsAPI** (`NEWS_API_KEY`)

Copy `.env.example` to `.env` and enter your keys:
```bash
cp .env.example .env
```
Edit `.env`:
```env
FINNHUB_API_KEY=your_finnhub_api_key_here
ALPHAVANTAGE_API_KEY=your_alphavantage_key_here
NEWS_API_KEY=your_newsapi_key_here
```
> [!NOTE]
> If no API keys are provided in `.env`, the collector gracefully falls back to live financial RSS feeds (Yahoo Finance & Google News RSS) with deduplication, ensuring the pipeline remains 100% executable without payment or API credentials.

---

## 🚀 Execution

### Run the Complete Pipeline
```bash
python run_analysis.py
```
This runs the end-to-end 8-step workflow:
1. **Data Ingestion & Chronological Split** (Train 2000–2017, Val 2018–2021, Test 2022–2026)
2. **Technical Feature Engineering & Pair Plot EDA** (87 factors)
3. **Financial News Collection** (Deduplicated articles to `data/financial_news.csv`)
4. **FinBERT NLP Sentiment Scoring** (Scored articles & daily aggregated features to CSV)
5. **Zero-Leakage Sentiment Merging** (Forward-fill, weekend roll-forward, lag-1 alignment)
6. **Model Training & Hyperparameter Tuning** (Baseline vs. Sentiment-Enhanced pipelines)
7. **PCA Dimensionality Reduction Experiments** (Fitted strictly on train data)
8. **Comparative Evaluation & Visualizations** (Saved to `outputs/figures/` and CSV summaries)

---

## 🧠 FinBERT Sentiment Architecture & Zero-Leakage Guarantee

- **Transformer Model**: Uses `ProsusAI/finbert` (Hugging Face / PyTorch), a BERT architecture specifically pre-trained and fine-tuned on financial text corpora (Financial PhraseBank).
- **Polarity Formula**: 
  $$\text{Sentiment Score} = P(\text{Positive}) - P(\text{Negative}) \in [-1, +1]$$
- **Daily Aggregated Features (9 features)**:
  - `Sentiment_Mean`, `Sentiment_Std`, `Sentiment_Pos_Ratio`, `Sentiment_Neg_Ratio`, `Sentiment_Neu_Ratio`
  - `News_Volume` (daily headline density)
  - `Sentiment_EMA_3`, `Sentiment_EMA_7` (short & intermediate exponential moving sentiment)
  - `Sentiment_Lag_1` (strictly lagged indicator preventing look-ahead bias)
- **Leakage Prevention**:
  1. Weekend / holiday news articles are stamped to the next valid NYSE trading session.
  2. Same-day sentiment is strictly shifted ($t-1$) before predicting day $t+1$ target close prices.
  3. Feature scalers (RobustScaler) and PCA are fitted strictly on the chronological training partition.

---

## 📊 Benchmark Leaderboard (Unseen Test Set: 2022–2026)

### Baseline vs. Sentiment-Enhanced Comparison

| Model | Feature Set | Test RMSE ($) | Test MAE ($) | Test $R^2$ | Directional Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Tuned Ensemble** | **Sentiment-Enhanced** | **$2.0234** | **$1.3445** | **0.9879** | **49.37%** |
| Tuned Ensemble | Baseline (Technical Only) | $2.0242 | $1.3442 | 0.9879 | 49.69% |
| **Tuned Lasso** | **Sentiment-Enhanced** | **$2.0242** | **$1.3388** | **0.9879** | **48.74%** |
| Tuned Lasso | Baseline (Technical Only) | $2.0242 | $1.3388 | 0.9879 | 48.74% |
| **PCA + Lasso** | **Baseline (Technical Only)** | **$2.0263** | **$1.3523** | **0.9879** | **48.95%** |
| PCA + Lasso | Sentiment-Enhanced | $2.0263 | $1.3523 | 0.9879 | 48.95% |
| **Tuned Random Forest** | **Sentiment-Enhanced** | **$2.0268** | **$1.3490** | **0.9879** | **47.59%** |
| Tuned Random Forest | Baseline (Technical Only) | $2.0313 | $1.3499 | 0.9878 | 48.43% |
| **Tuned XGBoost** | **Sentiment-Enhanced** | **$2.0273** | **$1.3456** | **0.9879** | **51.05%** |
| Tuned XGBoost | Baseline (Technical Only) | $2.0273 | $1.3453 | 0.9879 | 51.05% |
| **PCA + Ridge** | **Baseline (Technical Only)** | **$2.0287** | **$1.3564** | **0.9879** | **50.21%** |
| PCA + Ridge | Sentiment-Enhanced | $2.0287 | $1.3564 | 0.9879 | 50.10% |
| **PCA + Linear** | **Baseline (Technical Only)** | **$2.0291** | **$1.3569** | **0.9879** | **50.31%** |
| PCA + Linear | Sentiment-Enhanced | $2.0291 | $1.3569 | 0.9879 | 50.31% |
| **Lasso Regression** | **Sentiment-Enhanced** | **$2.0294** | **$1.3557** | **0.9879** | **50.21%** |
| Lasso Regression | Baseline (Technical Only) | $2.0294 | $1.3557 | 0.9879 | 50.21% |
| **Tuned Ridge** | **Sentiment-Enhanced** | **$2.0349** | **$1.3632** | **0.9878** | **50.21%** |
| Tuned Ridge | Baseline (Technical Only) | $2.0351 | $1.3635 | 0.9878 | 50.42% |
| **XGBoost (Default)** | **Baseline (Technical Only)** | **$2.0388** | **$1.3689** | **0.9877** | **48.53%** |
| **Random Forest (Default)** | **Baseline (Technical Only)** | **$2.0439** | **$1.3636** | **0.9877** | **50.21%** |
| Random Forest (Default) | Sentiment-Enhanced | $2.0448 | $1.3643 | 0.9877 | 50.10% |
| XGBoost (Default) | Sentiment-Enhanced | $2.0456 | $1.3745 | 0.9877 | 49.48% |
| **Ridge Regression** | **Sentiment-Enhanced** | **$2.0484** | **$1.3731** | **0.9876** | **50.42%** |
| Ridge Regression | Baseline (Technical Only) | $2.0493 | $1.3740 | 0.9876 | 50.31% |
| **Linear Regression** | **Baseline (Technical Only)** | **$2.2419** | **$1.5300** | **0.9852** | **48.64%** |
| Linear Regression | Sentiment-Enhanced | $2.3481 | $1.5878 | 0.9837 | 49.06% |

---

## 📈 Visualizations Generated

- `outputs/figures/sentiment_vs_stock_price.png`: Time-series overlay of rolling sentiment polarity, daily volume, and Close price.
- `outputs/figures/sentiment_distribution.png`: Histogram & KDE of FinBERT polarity score distribution across Positive, Negative, and Neutral classifications.
- `outputs/figures/sentiment_model_comparison.png`: Dual-panel comparative bar chart (RMSE & Directional Accuracy) across all models.
- `outputs/figures/eda_pair_plot.png`: Pairwise scatter & KDE relationships for key technical factors.
- `outputs/figures/pca_explained_variance.png`: Scree plot and cumulative explained variance curve for PCA components.
- `outputs/figures/best_model_predictions_test.png`: Actual vs. predicted test trajectory.
