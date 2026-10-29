import pandas as pd
import numpy as np
from typing import Tuple, List


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    stoch_k = 100 * ((close - low_min) / (high_max - low_min + 1e-9))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).fillna(0).cumsum()


def create_market_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    data = df.copy()
    
    # Price action & spreads
    data["HL_Spread"] = (data["High"] - data["Low"]) / (data["Close"] + 1e-9)
    data["CO_Spread"] = (data["Close"] - data["Open"]) / (data["Open"] + 1e-9)
    max_oc = np.maximum(data["Open"], data["Close"])
    min_oc = np.minimum(data["Open"], data["Close"])
    data["Upper_Shadow"] = (data["High"] - max_oc) / (data["Close"] + 1e-9)
    data["Lower_Shadow"] = (min_oc - data["Low"]) / (data["Close"] + 1e-9)
    data["Body_Spread"] = (data["Close"] - data["Open"]).abs() / (data["Close"] + 1e-9)
    
    for period in [1, 2, 3, 5, 10, 21]:
        data[f"Return_{period}d"] = data["Close"].pct_change(period)
        data[f"Log_Return_{period}d"] = np.log(data["Close"] / data["Close"].shift(period))

    # Trend & moving averages
    for w in [5, 10, 20, 50, 200]:
        data[f"SMA_{w}"] = data["Close"].rolling(window=w).mean()
        data[f"Close_to_SMA_{w}"] = (data["Close"] - data[f"SMA_{w}"]) / (data[f"SMA_{w}"] + 1e-9)
        
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]
    data["Golden_Cross"] = (data["SMA_50"] > data["SMA_200"]).astype(int)
    
    # Momentum oscillators
    data["RSI_14"] = compute_rsi(data["Close"], period=14)
    stoch_k, stoch_d = compute_stochastic_oscillator(data["High"], data["Low"], data["Close"], k_period=14, d_period=3)
    data["Stoch_K"] = stoch_k
    data["Stoch_D"] = stoch_d
    
    high_14 = data["High"].rolling(14).max()
    low_14 = data["Low"].rolling(14).min()
    data["Williams_R_14"] = -100 * ((high_14 - data["Close"]) / (high_14 - low_14 + 1e-9))
    
    for p in [5, 10, 21]:
        data[f"ROC_{p}"] = ((data["Close"] - data["Close"].shift(p)) / (data["Close"].shift(p) + 1e-9)) * 100

    # Volatility factors
    bb_mean = data["Close"].rolling(window=20).mean()
    bb_std = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = bb_mean + 2 * bb_std
    data["BB_Lower"] = bb_mean - 2 * bb_std
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / (bb_mean + 1e-9)
    data["BB_PctB"] = (data["Close"] - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"] + 1e-9)
    data["ATR_14"] = compute_atr(data["High"], data["Low"], data["Close"], period=14)
    data["NATR_14"] = data["ATR_14"] / (data["Close"] + 1e-9) * 100
    data["Vol_10d"] = data["Daily_Return"].rolling(window=10).std() * np.sqrt(252)
    data["Vol_20d"] = data["Daily_Return"].rolling(window=20).std() * np.sqrt(252)
    data["Vol_30d"] = data["Daily_Return"].rolling(window=30).std() * np.sqrt(252)

    # Volume factors
    data["OBV"] = compute_obv(data["Close"], data["Volume"])
    for v in [5, 10, 20]:
        data[f"Volume_SMA_{v}"] = data["Volume"].rolling(window=v).mean()
    data["Volume_Ratio_20"] = data["Volume"] / (data["Volume_SMA_20"] + 1e-9)
    data["Volume_ROC_5"] = data["Volume"].pct_change(5)
    
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    cum_pv = (typical_price * data["Volume"]).rolling(window=20).sum()
    cum_v = data["Volume"].rolling(window=20).sum()
    data["VWAP_20"] = cum_pv / (cum_v + 1e-9)
    data["Close_to_VWAP_20"] = (data["Close"] - data["VWAP_20"]) / (data["VWAP_20"] + 1e-9)

    # Lagged features
    for lag in [1, 2, 3, 5]:
        data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)
        data[f"Volume_Lag_{lag}"] = data["Volume"].shift(lag)
        data[f"Daily_Return_Lag_{lag}"] = data["Daily_Return"].shift(lag)
        data[f"HL_Spread_Lag_{lag}"] = data["HL_Spread"].shift(lag)

    # Calendar factors
    data["Day_of_Week"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month
    data["Quarter"] = data["Date"].dt.quarter
    data["Month_Sin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["Month_Cos"] = np.cos(2 * np.pi * data["Month"] / 12)
    data["DayOfWeek_Sin"] = np.sin(2 * np.pi * data["Day_of_Week"] / 5)
    data["DayOfWeek_Cos"] = np.cos(2 * np.pi * data["Day_of_Week"] / 5)

    # Target
    data["Target_Next_Close"] = data["Close"].shift(-1)
    data["Target_Next_Return"] = (data["Target_Next_Close"] - data["Close"]) / (data["Close"] + 1e-9)
    data["Target_Next_Direction"] = (data["Target_Next_Return"] > 0).astype(int)

    if drop_na:
        data = data.dropna().reset_index(drop=True)
        
    return data


def merge_sentiment_with_market_data(
    market_df: pd.DataFrame,
    daily_sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    df_market = market_df.copy()
    if df_market.empty:
        return df_market

    df_market["Date_Str"] = pd.to_datetime(df_market["Date"]).dt.strftime("%Y-%m-%d")
    trading_dates = sorted(df_market["Date_Str"].unique().tolist())
    trading_set = set(trading_dates)

    sentiment_cols = [
        "Sentiment_Mean", "Sentiment_Std", "Sentiment_Pos_Ratio",
        "Sentiment_Neg_Ratio", "Sentiment_Neu_Ratio", "News_Volume",
        "Sentiment_EMA_3", "Sentiment_EMA_7", "Sentiment_Lag_1"
    ]

    if daily_sentiment_df.empty or "Date" not in daily_sentiment_df.columns:
        for col in sentiment_cols:
            df_market[col] = 0.0
        return df_market.drop(columns=["Date_Str"])

    df_sent = daily_sentiment_df.copy()
    df_sent["Date_Str"] = pd.to_datetime(df_sent["Date"]).dt.strftime("%Y-%m-%d")

    trading_dates_dt = [pd.to_datetime(d) for d in trading_dates]

    def map_to_next_trading_day(dt_str: str) -> Optional[str]:
        if dt_str in trading_set:
            return dt_str
        dt = pd.to_datetime(dt_str)
        future_trading_days = [t for t in trading_dates_dt if t >= dt]
        if future_trading_days:
            return future_trading_days[0].strftime("%Y-%m-%d")
        return None

    df_sent["Trading_Date"] = df_sent["Date_Str"].apply(map_to_next_trading_day)
    df_sent = df_sent.dropna(subset=["Trading_Date"])

    if df_sent.empty:
        for col in sentiment_cols:
            df_market[col] = 0.0
        return df_market.drop(columns=["Date_Str"])

    agg_sent = df_sent.groupby("Trading_Date").agg({
        "Sentiment_Mean": "mean",
        "Sentiment_Std": "mean",
        "Sentiment_Pos_Ratio": "mean",
        "Sentiment_Neg_Ratio": "mean",
        "Sentiment_Neu_Ratio": "mean",
        "News_Volume": "sum",
    }).reset_index().rename(columns={"Trading_Date": "Date_Str"})

    merged = pd.merge(df_market, agg_sent, on="Date_Str", how="left")

    merged["News_Volume"] = merged["News_Volume"].fillna(0)
    for col in ["Sentiment_Mean", "Sentiment_Std", "Sentiment_Pos_Ratio", "Sentiment_Neg_Ratio", "Sentiment_Neu_Ratio"]:
        merged[col] = merged[col].fillna(0.0)

    merged["Sentiment_EMA_3"] = merged["Sentiment_Mean"].ewm(span=3, adjust=False).mean()
    merged["Sentiment_EMA_7"] = merged["Sentiment_Mean"].ewm(span=7, adjust=False).mean()
    merged["Sentiment_Lag_1"] = merged["Sentiment_Mean"].shift(1).fillna(0.0)

    merged = merged.drop(columns=["Date_Str"]).reset_index(drop=True)
    return merged


def get_feature_columns(df: pd.DataFrame, include_sentiment: bool = True) -> List[str]:
    exclude_cols = ["Date", "Target_Next_Close", "Target_Next_Return", "Target_Next_Direction"]
    sentiment_cols = [
        "Sentiment_Mean", "Sentiment_Std", "Sentiment_Pos_Ratio",
        "Sentiment_Neg_Ratio", "Sentiment_Neu_Ratio", "News_Volume",
        "Sentiment_EMA_3", "Sentiment_EMA_7", "Sentiment_Lag_1"
    ]
    if not include_sentiment:
        return [col for col in df.columns if col not in exclude_cols and col not in sentiment_cols]
    return [col for col in df.columns if col not in exclude_cols]

