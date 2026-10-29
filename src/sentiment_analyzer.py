import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"


class FinBERTSentimentAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FinBERT] Initializing {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.labels = [self.model.config.id2label[i].capitalize() for i in range(len(self.model.config.id2label))]

    def analyze_texts(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        results = []
        if not texts:
            return results

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = [str(t) if t else "" for t in texts[i : i + batch_size]]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

                for prob in probs:
                    pos_idx = 0
                    neg_idx = 1
                    neu_idx = 2
                    for idx, lab in enumerate(self.labels):
                        if lab.lower() == "positive":
                            pos_idx = idx
                        elif lab.lower() == "negative":
                            neg_idx = idx
                        elif lab.lower() == "neutral":
                            neu_idx = idx

                    p_pos = float(prob[pos_idx])
                    p_neg = float(prob[neg_idx])
                    p_neu = float(prob[neu_idx])
                    pred_label = self.labels[int(np.argmax(prob))]
                    sentiment_score = float(p_pos - p_neg)

                    results.append({
                        "Sentiment_Label": pred_label,
                        "Score_Positive": p_pos,
                        "Score_Negative": p_neg,
                        "Score_Neutral": p_neu,
                        "Sentiment_Score": sentiment_score,
                    })

        return results


def run_sentiment_analysis(
    news_df: pd.DataFrame,
    classified_save_path: str = "data/financial_news_sentiment.csv",
    daily_save_path: str = "data/daily_sentiment.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if news_df.empty or "Headline" not in news_df.columns:
        empty_classified = pd.DataFrame(columns=["Date", "Headline", "Sentiment_Label", "Score_Positive", "Score_Negative", "Score_Neutral", "Sentiment_Score"])
        empty_daily = pd.DataFrame(columns=["Date", "Sentiment_Mean", "Sentiment_Std", "Sentiment_Pos_Ratio", "Sentiment_Neg_Ratio", "Sentiment_Neu_Ratio", "News_Volume", "Sentiment_EMA_3", "Sentiment_EMA_7", "Sentiment_Lag_1"])
        return empty_classified, empty_daily

    analyzer = FinBERTSentimentAnalyzer()
    
    texts_to_score = [
        f"{row['Headline']}. {row['Summary']}".strip() if pd.notna(row.get("Summary")) and row.get("Summary") else str(row["Headline"])
        for _, row in news_df.iterrows()
    ]
    
    print(f"[FinBERT] Scoring {len(texts_to_score)} articles...")
    scores = analyzer.analyze_texts(texts_to_score, batch_size=32)
    scores_df = pd.DataFrame(scores)

    classified_df = pd.concat([news_df.reset_index(drop=True), scores_df], axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(classified_save_path)), exist_ok=True)
    classified_df.to_csv(classified_save_path, index=False)
    print(f"[OK] Saved {len(classified_df)} classified news articles to {classified_save_path}")

    # Daily Aggregation
    classified_df["Date"] = pd.to_datetime(classified_df["Date"]).dt.strftime("%Y-%m-%d")
    
    daily_records = []
    for dt, group in classified_df.groupby("Date"):
        n_items = len(group)
        pos_count = (group["Sentiment_Label"] == "Positive").sum()
        neg_count = (group["Sentiment_Label"] == "Negative").sum()
        neu_count = (group["Sentiment_Label"] == "Neutral").sum()
        
        daily_records.append({
            "Date": dt,
            "Sentiment_Mean": float(group["Sentiment_Score"].mean()),
            "Sentiment_Std": float(group["Sentiment_Score"].std() if n_items > 1 else 0.0),
            "Sentiment_Pos_Ratio": float(pos_count / n_items),
            "Sentiment_Neg_Ratio": float(neg_count / n_items),
            "Sentiment_Neu_Ratio": float(neu_count / n_items),
            "News_Volume": int(n_items),
        })

    daily_df = pd.DataFrame(daily_records).sort_values("Date").reset_index(drop=True)
    daily_df["Sentiment_EMA_3"] = daily_df["Sentiment_Mean"].ewm(span=3, adjust=False).mean()
    daily_df["Sentiment_EMA_7"] = daily_df["Sentiment_Mean"].ewm(span=7, adjust=False).mean()
    daily_df["Sentiment_Lag_1"] = daily_df["Sentiment_Mean"].shift(1).fillna(0.0)

    os.makedirs(os.path.dirname(os.path.abspath(daily_save_path)), exist_ok=True)
    daily_df.to_csv(daily_save_path, index=False)
    print(f"[OK] Saved {len(daily_df)} daily aggregated sentiment records to {daily_save_path}")

    return classified_df, daily_df
