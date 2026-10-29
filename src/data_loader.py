import pandas as pd
import numpy as np
from typing import Dict, Any


def parse_mixed_dates(date_series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(date_series, format="mixed", dayfirst=False)
    except Exception:
        formats = ["%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y"]
        parsed = pd.Series(index=date_series.index, dtype="datetime64[ns]")
        for fmt in formats:
            mask = parsed.isna()
            if not mask.any():
                break
            try:
                parsed[mask] = pd.to_datetime(date_series[mask], format=fmt, errors="coerce")
            except Exception:
                pass
        mask = parsed.isna()
        if mask.any():
            parsed[mask] = pd.to_datetime(date_series[mask], errors="coerce")
        return parsed


def load_stock_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = [col.strip().capitalize() if col.strip().lower() in ['date', 'close', 'high', 'low', 'open', 'volume'] else col.strip() for col in df.columns]
    df["Date"] = parse_mixed_dates(df["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def validate_stock_data(df: pd.DataFrame) -> Dict[str, Any]:
    high_low_violation = (df["High"] < df["Low"]).sum()
    high_close_violation = (df["High"] < df["Close"]).sum()
    low_close_violation = (df["Low"] > df["Close"]).sum()
    
    return {
        "Total_Rows": len(df),
        "Date_Range": f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
        "Start_Date": df["Date"].min().strftime("%Y-%m-%d"),
        "End_Date": df["Date"].max().strftime("%Y-%m-%d"),
        "Missing_Values": df.isnull().sum().to_dict(),
        "Price_Logic_Pass": bool(high_low_violation == 0 and high_close_violation == 0 and low_close_violation == 0),
        "Min_Close": float(df["Close"].min()),
        "Max_Close": float(df["Close"].max()),
        "Mean_Close": float(df["Close"].mean()),
        "Std_Close": float(df["Close"].std()),
        "Min_Volume": int(df["Volume"].min()),
        "Max_Volume": int(df["Volume"].max()),
    }
