import streamlit as st
import pandas as pd
import pandas_ta as ta
import requests
import time

st.set_page_config(page_title="Quant Signal Pro", layout="wide")

REFRESH_SEC = 60
TOP_N = 20
BINANCE_API = "https://api.binance.com/api/v3"

# -----------------------------
# SYMBOL LIST
# -----------------------------
def get_symbols():
    try:
        res = requests.get(f"{BINANCE_API}/ticker/24hr", timeout=10)
        data = res.json()

        # Binance hata döndürürse
        if not isinstance(data, list):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df[df["symbol"].str.endswith("USDT")]
        df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce")
        df = df.sort_values("quoteVolume", ascending=False)

        return df.head(100)

    except:
        return pd.DataFrame()


# -----------------------------
# KLINES
# -----------------------------
def get_klines(symbol, interval="15m", limit=200):
    try:
        url = f"{BINANCE_API}/klines?symbol={symbol}&interval={interval}&limit={limit}"
        data = requests.get(url, timeout=10).json()

        if not isinstance(data, list):
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.iloc[:, :6]
        df.columns = ["ts","open","high","low","close","volume"]

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        return df

    except:
        return pd.DataFrame()


# -----------------------------
# ANALYSIS
# -----------------------------
def analyze(symbol):
    d15 = get_klines(symbol, "15m")
    d4h = get_klines(symbol, "4h")

    if d15.empty or d4h.empty:
        return None

    d15.ta.rsi(length=14, append=True)
    d15.ta.macd(append=True)
    d4h.ta.ema(length=200, append=True)

    l15 = d15.iloc[-1]
    l4 = d4h.iloc[-1]

    score = 0

    if "EMA_200" in l4 and l4["close"] > l4["EMA_200"]:
        score += 40
    else:
        score -= 40

    if "RSI_14" in l15:
        if l15["RSI_14"] > 60:
            score += 20
        elif l15["RSI_14"] < 40:
            score -= 20

    if "MACDh_12_26_9" in l15:
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


# -----------------------------
# UI
# -----------------------------
st.title("⚡ Quant Signal Pro")
st.caption("Top güçlü sinyaller")

symbols = get_symbols()

if symbols.empty:
    st.error("Binance verisi alınamadı. Biraz sonra tekrar deneyin.")
    st.stop()

results = []

for sym in symbols["symbol"].values:
    result = analyze(sym)
    if result:
        sig, score, price = result
        results.append([sym, sig, score, price])

if len(results) == 0:
    st.warning("Henüz sinyal üretilemedi.")
    st.stop()

df = pd.DataFrame(results, columns=["Coin","Sinyal","Skor","Fiyat"])
df = df.sort_values("Skor", ascending=False)

st.dataframe(df.head(TOP_N), use_container_width=True)

time.sleep(REFRESH_SEC)
st.rerun()
