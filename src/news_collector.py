import os
import re
import json
import logging
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def clean_headline(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_finnhub_news(ticker: str, api_key: str, from_date: str = "", to_date: str = "") -> List[Dict[str, Any]]:
    articles = []
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. Company news
    if from_date and to_date:
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={api_key}"
    else:
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&token={api_key}"
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list):
                for item in data:
                    ts = item.get("datetime", 0)
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else ""
                    headline = clean_headline(item.get("headline", ""))
                    if headline:
                        articles.append({
                            "Date": dt,
                            "Headline": headline,
                            "Summary": clean_headline(item.get("summary", "")),
                            "Source": item.get("source", "Finnhub"),
                            "URL": item.get("url", ""),
                        })
    except Exception as e:
        logger.warning(f"Finnhub company news error: {e}")

    # 2. General market news
    try:
        gen_url = f"https://finnhub.io/api/v1/news?category=general&token={api_key}"
        req = urllib.request.Request(gen_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list):
                for item in data:
                    ts = item.get("datetime", 0)
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else ""
                    headline = clean_headline(item.get("headline", ""))
                    if headline:
                        articles.append({
                            "Date": dt,
                            "Headline": headline,
                            "Summary": clean_headline(item.get("summary", "")),
                            "Source": item.get("source", "Finnhub-General"),
                            "URL": item.get("url", ""),
                        })
    except Exception as e:
        logger.warning(f"Finnhub general news error: {e}")

    return articles


def fetch_alphavantage_news(ticker: str, api_key: str) -> List[Dict[str, Any]]:
    articles = []
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}&limit=50"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            feed = data.get("feed", [])
            for item in feed:
                time_published = item.get("time_published", "")
                dt = time_published[:8]
                if len(dt) == 8:
                    dt = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}"
                else:
                    dt = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                headline = clean_headline(item.get("title", ""))
                if headline:
                    articles.append({
                        "Date": dt,
                        "Headline": headline,
                        "Summary": clean_headline(item.get("summary", "")),
                        "Source": item.get("source", "AlphaVantage"),
                        "URL": item.get("url", ""),
                    })
    except Exception as e:
        logger.warning(f"Alpha Vantage news error: {e}")
    return articles


def fetch_rss_news(ticker: str = "AAPL") -> List[Dict[str, Any]]:
    articles = []
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. Yahoo Finance RSS
    yahoo_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        req = urllib.request.Request(yahoo_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            root = ET.fromstring(resp.read())
            for item in root.findall(".//item"):
                title = item.find("title")
                pub_date = item.find("pubDate")
                link = item.find("link")
                desc = item.find("description")
                
                headline = clean_headline(title.text if title is not None else "")
                dt_str = ""
                if pub_date is not None and pub_date.text:
                    try:
                        parsed_dt = pd.to_datetime(pub_date.text)
                        dt_str = parsed_dt.strftime("%Y-%m-%d")
                    except Exception:
                        dt_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                else:
                    dt_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    
                if headline:
                    articles.append({
                        "Date": dt_str,
                        "Headline": headline,
                        "Summary": clean_headline(desc.text if desc is not None else ""),
                        "Source": "Yahoo Finance RSS",
                        "URL": link.text if link is not None else "",
                    })
    except Exception as e:
        logger.warning(f"Yahoo Finance RSS error: {e}")

    # 2. Google News Financial RSS
    google_url = f"https://news.google.com/rss/search?q={ticker}+stock+market+economy+finance&hl=en-US&gl=US&ceid=US:en"
    try:
        req = urllib.request.Request(google_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            root = ET.fromstring(resp.read())
            for item in root.findall(".//item"):
                title = item.find("title")
                pub_date = item.find("pubDate")
                link = item.find("link")
                headline = clean_headline(title.text if title is not None else "")
                dt_str = ""
                if pub_date is not None and pub_date.text:
                    try:
                        parsed_dt = pd.to_datetime(pub_date.text)
                        dt_str = parsed_dt.strftime("%Y-%m-%d")
                    except Exception:
                        dt_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                else:
                    dt_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

                if headline:
                    articles.append({
                        "Date": dt_str,
                        "Headline": headline,
                        "Summary": "",
                        "Source": "Google News RSS",
                        "URL": link.text if link is not None else "",
                    })
    except Exception as e:
        logger.warning(f"Google News RSS error: {e}")

    return articles


def collect_financial_news(
    ticker: Optional[str] = None,
    save_path: str = "data/financial_news.csv",
    force_refresh: bool = False
) -> pd.DataFrame:
    ticker = ticker or os.getenv("STOCK_TICKER", "AAPL")
    
    if os.path.exists(save_path) and not force_refresh:
        try:
            cached_df = pd.read_csv(save_path)
            if not cached_df.empty and "Headline" in cached_df.columns:
                print(f"[OK] Loaded {len(cached_df)} existing financial news articles from {save_path}")
                return cached_df
        except Exception:
            pass

    finnhub_key = os.getenv("FINNHUB_API_KEY", "").strip()
    alpha_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    newsapi_key = os.getenv("NEWS_API_KEY", "").strip()

    all_articles = []

    if finnhub_key:
        print(f"[API] Collecting news from Finnhub API for ticker: {ticker}...")
        fh_articles = fetch_finnhub_news(ticker, finnhub_key)
        all_articles.extend(fh_articles)
        print(f"  Retrieved {len(fh_articles)} articles from Finnhub.")
    
    if alpha_key:
        print(f"[API] Collecting news from Alpha Vantage for ticker: {ticker}...")
        av_articles = fetch_alphavantage_news(ticker, alpha_key)
        all_articles.extend(av_articles)
        print(f"  Retrieved {len(av_articles)} articles from Alpha Vantage.")

    if not finnhub_key and not alpha_key:
        print("[CONFIG] No financial news API key detected in .env.")
        print("         To enable custom API queries, add FINNHUB_API_KEY to your .env file.")
        print(f"[LIVE] Fetching real-time market news for {ticker} via live financial feeds...")
        rss_articles = fetch_rss_news(ticker)
        all_articles.extend(rss_articles)
        print(f"  Retrieved {len(rss_articles)} real-time market articles.")

    if not all_articles and os.path.exists(save_path):
        return pd.read_csv(save_path)

    df = pd.DataFrame(all_articles)
    if df.empty:
        df = pd.DataFrame(columns=["Date", "Headline", "Summary", "Source", "URL"])
    else:
        # Deduplication
        df["Headline_Norm"] = df["Headline"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
        df = df.drop_duplicates(subset=["Date", "Headline_Norm"]).drop(columns=["Headline_Norm"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["Date", "Headline"]).sort_values("Date").reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[OK] Saved {len(df)} sanitized, deduplicated news articles to {save_path}")
    return df
