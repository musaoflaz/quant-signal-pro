import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import pandas_ta as ta
import time
from datetime import datetime

# =========================
# AYARLAR
# =========================
st.set_page_config(page_title="Quant Signal Pro", layout="wide")

REFRESH_SEC = 60          # sayfa yenileme aralığı
UNIVERSE_LIMIT = 200      # kaç coini tarayalım (ilk N, hacme göre)
TOP_N = 20                # ekranda gösterilecek top N

# =========================
# BİNANCE (SPOT) BAĞLANTISI
# =========================
# NOT: Streamlit Cloud'da futures bazen bloklanıyor. Bu yüzden SPOT kullanıyoruz.
exchange = ccxt.binance({"enableRateLimit": True})

@st.cache_data(ttl=300)
def get_top_symbols_by_volume(limit=200):
    """
    USDT spot paritelerinde, hacme göre en yüksek ilk 'limit' sembol.
    """
    tickers = exchange.fetch_tickers()
    markets = exchange.load_markets()

    rows = []
    for sym, m in markets.items():
        # SPOT USDT pariteleri
        if m.get("spot") and m.get("active") and m.get("quote") == "USDT":
            t = tickers.get(sym, {})
            last = t.get("last", None)
            quote_vol = t.get("quoteVolume", 0) or 0
            if last is None:
                continue
            rows.append((sym, float(quote_vol), float(last)))

    df = pd.DataFrame(rows, columns=["symbol", "quoteVolume", "last"])
    df = df.sort_values("quoteVolume", ascending=False).head(limit).reset_index(drop=True)
    return df

def fetch_ohlcv(symbol, timeframe="15m", limit=250):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or len(df) < 210:
        return df

    df = df.copy()
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.adx(length=14, append=True)

    # hacim oranı (opsiyonel)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    return df

def signal_score(d15: pd.DataFrame, d4h: pd.DataFrame):
    """
    Basit ama stabil bir puanlama.
    - H4 EMA200 trend filtresi (ana yön)
    - 15m RSI, MACD histogram, ADX (trend gücü)
    - Vol_ratio (breakout filtresi)
    """
    if d15 is None or d4h is None or d15.empty or d4h.empty:
        return "WAIT", 0, "No data"

    l15 = d15.iloc[-1]
    l4 = d4h.iloc[-1]

    score = 0
    reasons = []

    # 1) Ana trend (H4)
    if "EMA_200" in d4h.columns and float(l4["close"]) > float(l4["EMA_200"]):
        score += 35
        reasons.append("H4>EMA200 (Boğa)")
    else:
        score -= 35
        reasons.append("H4<EMA200 (Ayı)")

    # 2) RSI (15m)
    rsi = float(l15.get("RSI_14", 50))
    if rsi >= 60:
        score += 20
        reasons.append("RSI güçlü")
    elif rsi <= 40:
        score -= 20
        reasons.append("RSI zayıf")
    else:
        reasons.append("RSI nötr")

    # 3) MACD histogram (15m)
    macdh = float(l15.get("MACDh_12_26_9", 0))
    if macdh > 0:
        score += 15
        reasons.append("MACD+")
    else:
        score -= 15
        reasons.append("MACD-")

    # 4) ADX (15m) trend gücü
    adx = float(l15.get("ADX_14", 0))
    if adx >= 25:
        score += 10
        reasons.append("ADX trend")
    else:
        reasons.append("ADX düşük")

    # 5) Hacim oranı (15m)
    vr = float(l15.get("vol_ratio", 1))
    if vr >= 1.3:
        score += 5
        reasons.append("Hacim↑")
    elif vr <= 0.8:
        score -= 5
        reasons.append("Hacim↓")

    score = int(np.clip(score, -100, 100))

    # Sinyal etiketi
    if score >= 60:
        sig = "GÜÇLÜ LONG"
    elif score >= 25:
        sig = "LONG"
    elif score <= -60:
        sig = "GÜÇLÜ SHORT"
    elif score <= -25:
        sig = "SHORT"
    else:
        sig = "BEKLE"

    return sig, score, " | ".join(reasons)

# =========================
# UI
# =========================
st.title("⚡ Quant Signal Pro (SPOT)")
st.caption("Sade ekran: En güçlü 20 Long/Short sinyali (hacme göre tarama).")

colA, colB, colC = st.columns([1.2, 1, 1])
with colA:
    st.write(f"🕒 Son güncelleme: **{datetime.now().strftime('%H:%M:%S')}**")
with colB:
    universe_limit = st.slider("Tarama (ilk N coin)", 50, 400, UNIVERSE_LIMIT, 50)
with colC:
    top_n = st.slider
