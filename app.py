import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

st.set_page_config(page_title="Quant Signal Pro", page_icon="⚡", layout="wide")

st.markdown("## ⚡ Quant Signal Pro")
st.caption("Ultra Stable Engine")

TOP_N = st.selectbox("Top", [10, 20, 30, 50], index=1)
REFRESH = st.selectbox("Yenile (sn)", [30, 60, 120], index=1)

COINGECKO_MARKETS = "https://api.coingecko.com/api/v3/coins/markets"
COINGECKO_CHART = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

@st.cache_data(ttl=120)
def get_top_coins(limit=50):
    params = {
        "vs_currency": "usd",
        "order": "volume_desc",
        "per_page": limit,
        "page": 1
    }
    r = requests.get(COINGECKO_MARKETS, params=params)
    return r.json()

@st.cache_data(ttl=120)
def get_chart(coin_id):
    params = {
        "vs_currency": "usd",
        "days": "2",
        "interval": "hourly"
    }
    r = requests.get(COINGECKO_CHART.format(id=coin_id), params=params)
    data = r.json()

    prices = data.get("prices", [])
    df = pd.DataFrame(prices, columns=["ts", "price"])
    df["price"] = pd.to_numeric(df["price"])
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series, period=50):
    return series.ewm(span=period, adjust=False).mean()

def analyze(df):
    df["EMA50"] = calculate_ema(df["price"], 50)
    df["RSI"] = calculate_rsi(df["price"], 14)

    last = df.iloc[-1]

    score = 0

    if last["price"] > last["EMA50"]:
        score += 40
    else:
        score -= 40

    if last["RSI"] > 60:
        score += 20
    elif last["RSI"] < 40:
        score -= 20

    if score >= 60:
        signal = "🚀 GÜÇLÜ LONG"
    elif score >= 20:
        signal = "🟢 LONG"
    elif score <= -60:
        signal = "🔥 GÜÇLÜ SHORT"
    elif score <= -20:
        signal = "🔴 SHORT"
    else:
        signal = "⚪ BEKLE"

    return signal, score, last["price"]

coins = get_top_coins(50)

results = []

for coin in coins:
    try:
        df = get_chart(coin["id"])
        if len(df) < 60:
            continue
        sig, score, price = analyze(df)
        results.append([
            coin["symbol"].upper(),
            sig,
            score,
            round(price, 4)
        ])
    except:
        continue

if len(results) == 0:
    st.error("Veri alınamadı. CoinGecko rate limit olabilir.")
else:
    df = pd.DataFrame(results, columns=["Coin", "Sinyal", "Skor", "Fiyat"])
    df = df.sort_values("Skor", ascending=False).head(TOP_N)
    st.dataframe(df, use_container_width=True)

st.info(f"Güncelleme: {datetime.now().strftime('%H:%M:%S')}")

time.sleep(REFRESH)
st.rerun()
