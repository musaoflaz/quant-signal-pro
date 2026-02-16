import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
import time
from datetime import datetime

st.set_page_config(page_title="Quant Signal Pro", page_icon="⚡", layout="wide")

st.markdown("## ⚡ Quant Signal Pro")
st.caption("Global veri sağlayıcı (Binance blok bypass)")

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
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def analyze(df):
    df["price"] = pd.to_numeric(df["price"])

    df.ta.rsi(close="price", length=14, append=True)
    df.ta.ema(close="price", length=50, append=True)

    last = df.iloc[-1]

    score = 0

    if last["price"] > last["EMA_50"]:
        score += 40
    else:
        score -= 40

    if last["RSI_14"] > 60:
        score += 20
    elif last["RSI_14"] < 40:
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
        sig, score, price = analyze(df)
        results.append([
            coin["symbol"].upper(),
            sig,
            score,
            round(price, 4)
        ])
    except:
        continue

df = pd.DataFrame(results, columns=["Coin", "Sinyal", "Skor", "Fiyat"])
df = df.sort_values("Skor", ascending=False).head(TOP_N)

st.dataframe(df, use_container_width=True)

st.info(f"Güncelleme: {datetime.now().strftime('%H:%M:%S')}")

time.sleep(REFRESH)
st.rerun()
