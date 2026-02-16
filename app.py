import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import pandas_ta as ta
import plotly.graph_objects as go
import requests
import time
from datetime import datetime

st.set_page_config(page_title="Quant Signal Pro", layout="wide")

# =========================
# TELEGRAM AYARI
# =========================
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""

def send_telegram(msg):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

# =========================
# BINANCE ENGINE
# =========================
exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "future"}
})

@st.cache_data(ttl=300)
def get_symbols():
    markets = exchange.load_markets()
    symbols = []
    for s, m in markets.items():
        if m.get("linear") and m.get("quote") == "USDT" and m.get("active"):
            symbols.append(s)
    return symbols[:150]

def get_data(symbol, tf):
    ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=200)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def add_indicators(df):
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.adx(length=14, append=True)
    return df

def score_signal(df15, df1h):
    if df15.empty or df1h.empty:
        return "WAIT", 0

    l15 = df15.iloc[-1]
    l1h = df1h.iloc[-1]

    score = 0

    if l1h["close"] > l1h["EMA_200"]:
        score += 30
    else:
        score -= 30

    if l15["RSI_14"] > 60:
        score += 20
    elif l15["RSI_14"] < 40:
        score -= 20

    if l15["MACDh_12_26_9"] > 0:
        score += 15
    else:
        score -= 15

    if score >= 40:
        return "LONG", score
    elif score <= -40:
        return "SHORT", score
    else:
        return "WAIT", score

# =========================
# UI
# =========================
st.title("⚡ Quant Signal Pro")
st.caption("Top 20 güçlü sinyaller")

symbols = get_symbols()
results = []

progress = st.progress(0)

for i, sym in enumerate(symbols):
    try:
        d15 = add_indicators(get_data(sym, "15m"))
        d1h = add_indicators(get_data(sym, "1h"))
        sig, sc = score_signal(d15, d1h)

        if sig != "WAIT":
            results.append({
                "Coin": sym,
                "Signal": sig,
                "Score": sc,
                "Price": round(d15.iloc[-1]["close"], 4)
            })

        progress.progress((i+1)/len(symbols))
    except:
        continue

df = pd.DataFrame(results)
if not df.empty:
    df["Rank"] = df["Score"].abs()
    top20 = df.sort_values("Rank", ascending=False).head(20)
    st.dataframe(top20[["Coin","Signal","Score","Price"]], use_container_width=True)
else:
    st.write("Sinyal yok.")

# =========================
# AUTO REFRESH
# =========================
time.sleep(20)
st.rerun()
