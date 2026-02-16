import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
import time
from datetime import datetime

st.set_page_config(page_title="Quant Signal Pro", layout="wide")

REFRESH_SEC = 60
TOP_N = 20

BINANCE_API = "https://api.binance.com/api/v3"

def get_symbols():
    res = requests.get(f"{BINANCE_API}/ticker/24hr")
    data = res.json()
    df = pd.DataFrame(data)
    df = df[df["symbol"].str.endswith("USDT")]
    df["quoteVolume"] = pd.to_numeric(df["quoteVolume"])
    df = df.sort_values("quoteVolume", ascending=False)
    return df.head(200)

def get_klines(symbol, interval="15m", limit=200):
    url = f"{BINANCE_API}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(data)
    df = df.iloc[:, :6]
    df.columns = ["ts","open","high","low","close","volume"]
    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])
    return df

def analyze(symbol):
    d15 = get_klines(symbol, "15m")
    d4h = get_klines(symbol, "4h")

    d15.ta.rsi(length=14, append=True)
    d15.ta.macd(append=True)
    d4h.ta.ema(length=200, append=True)

    l15 = d15.iloc[-1]
    l4 = d4h.iloc[-1]

    score = 0

    if l4["close"] > l4["EMA_200"]:
        score += 40
    else:
        score -= 40

    if l15["RSI_14"] > 60:
        score += 20
    elif l15["RSI_14"] < 40:
        score -= 20

    if l15["MACDh_12_26_9"] > 0:
        score += 20
    else:
        score -= 20

    if score >= 60:
        signal = "GÜÇLÜ LONG"
    elif score >= 20:
        signal = "LONG"
    elif score <= -60:
        signal = "GÜÇLÜ SHORT"
    elif score <= -20:
        signal = "SHORT"
    else:
        signal = "BEKLE"

    return signal, score, l15["close"]

st.title("⚡ Quant Signal Pro")
st.caption("Top güçlü sinyaller")

symbols = get_symbols()

results = []

for sym in symbols["symbol"].values[:100]:
    try:
        sig, score, price = analyze(sym)
        results.append([sym, sig, score, price])
    except:
        continue

df = pd.DataFrame(results, columns=["Coin","Sinyal","Skor","Fiyat"])
df = df.sort_values("Skor", ascending=False)

st.dataframe(df.head(TOP_N), use_container_width=True)

time.sleep(REFRESH_SEC)
st.rerun()
