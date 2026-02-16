import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
import time
import random
from datetime import datetime

# =========================
# PRO SETTINGS (Sade UI)
# =========================
st.set_page_config(page_title="Quant Signal Pro", page_icon="⚡", layout="wide")

# Sade başlık
st.markdown("## ⚡ Quant Signal Pro")
st.caption("Top güçlü sinyaller (fallback + retry + cache)")

# Kontroller (karmasık değil)
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    TOP_N = st.selectbox("Top", [10, 20, 30, 50], index=1)
with c2:
    REFRESH_SEC = st.selectbox("Yenile", [30, 60, 120, 300], index=1)
with c3:
    universe_n = st.selectbox("Evren", [50, 100, 150, 200], index=1)
with c4:
    only_strong = st.toggle("Sadece GÜÇLÜ sinyaller", value=False)

# =========================
# BINANCE PRO CLIENT
# =========================
BINANCE_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]
API_V3 = "/api/v3"

# Session + headers
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (QuantSignalPro; Streamlit)",
    "Accept": "application/json",
})

def _sleep_jitter(min_s=0.15, max_s=0.45):
    time.sleep(random.uniform(min_s, max_s))

def safe_get_json(path: str, params: dict | None = None, timeout=12, max_tries=4):
    """
    Çoklu endpoint + retry + backoff.
    Binance bazen error JSON döndürür (dict). Biz bunu yakalayıp None döndürürüz.
    """
    last_err = None

    # endpointleri karıştırarak dene (hepsi aynı anda rate limit yemesin)
    bases = BINANCE_BASES[:]
    random.shuffle(bases)

    for base in bases:
        url = f"{base}{API_V3}{path}"
        for attempt in range(1, max_tries + 1):
            try:
                r = SESSION.get(url, params=params, timeout=timeout)
                if r.status_code == 200:
                    data = r.json()
                    # Binance error formatı: {"code":..., "msg":...}
                    if isinstance(data, dict) and ("code" in data and "msg" in data):
                        last_err = f"Binance error: {data.get('code')} {data.get('msg')}"
                        break
                    return data
                else:
                    last_err = f"HTTP {r.status_code}: {r.text[:180]}"
            except Exception as e:
                last_err = str(e)

            # exponential backoff + jitter
            backoff = (2 ** (attempt - 1)) * 0.6 + random.uniform(0, 0.25)
            time.sleep(backoff)

        # endpoint değişmeden önce küçük jitter
        _sleep_jitter()

    return None, last_err

@st.cache_data(ttl=120)
def get_symbols_top_usdt(universe_n=100):
    """
    24hr ticker -> hacme göre en likit USDT pariteleri
    """
    data = safe_get_json("/ticker/24hr")
    if isinstance(data, tuple):
        data, err = data
    else:
        err = None

    if not isinstance(data, list):
        return None, err or "Ticker verisi list değil (API engeli olabilir)."

    df = pd.DataFrame(data)
    if "symbol" not in df.columns or "quoteVolume" not in df.columns:
        return None, "Ticker kolonları eksik."

    df = df[df["symbol"].astype(str).str.endswith("USDT")].copy()
    # bazı stable-stable pariteleri istersen çıkar
    exclude = {"BUSDUSDT", "TUSDUSDT", "USDCUSDT", "FDUSDUSDT", "USDPUSDT", "EURUSDT"}
    df = df[~df["symbol"].isin(exclude)]

    df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce").fillna(0)
    df = df.sort_values("quoteVolume", ascending=False).head(int(universe_n))
    return df, None

@st.cache_data(ttl=120)
def get_klines(symbol, interval="15m", limit=220):
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    data = safe_get_json("/klines", params=params)
    if isinstance(data, tuple):
        data, err = data
    else:
        err = None

    if not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], list):
        return None, err or f"Klines bozuk ({symbol}, {interval})."

    # Binance kline: [openTime, open, high, low, close, volume, ...]
    df = pd.DataFrame(data, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "tbbav", "tbqav", "ignore"
    ])
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    if len(df) < 210:
        return None, f"Yetersiz mum sayısı ({symbol}, {interval})."
    return df, None

def score_signal(d15: pd.DataFrame, d4h: pd.DataFrame):
    """
    Sinyal: Trend filtresi (H4 EMA200) + Momentum (RSI) + MACD histogram
    Basit ama sağlam. İstersen sonra ADX/BB/ATR ekleriz.
    """
    # indikatörler
    d15 = d15.copy()
    d4h = d4h.copy()

    d15.ta.rsi(length=14, append=True)
    d15.ta.macd(append=True)  # MACDh_12_26_9
    d4h.ta.ema(length=200, append=True)

    l15 = d15.iloc[-1]
    l4  = d4h.iloc[-1]

    score = 0
    reasons = []

    # Trend filtresi
    if l4["close"] > l4["EMA_200"]:
        score += 45
        reasons.append("H4>EMA200 (Boğa)")
    else:
        score -= 45
        reasons.append("H4<EMA200 (Ayı)")

    # RSI momentum
    rsi = float(l15.get("RSI_14", 50))
    if rsi >= 65:
        score += 20
        reasons.append("RSI güçlü +")
    elif rsi >= 55:
        score += 10
        reasons.append("RSI +")
    elif rsi <= 35:
        score -= 20
        reasons.append("RSI güçlü -")
    elif rsi <= 45:
        score -= 10
        reasons.append("RSI -")
    else:
        reasons.append("RSI nötr")

    # MACD histogram
    macdh = float(l15.get("MACDh_12_26_9", 0))
    if macdh > 0:
        score += 20
        reasons.append("MACD +")
    else:
        score -= 20
        reasons.append("MACD -")

    # Kırp
    score = int(np.clip(score, -100, 100))

    if score >= 70:
        sig = "🚀 GÜÇLÜ LONG"
    elif score >= 30:
        sig = "🟢 LONG"
    elif score <= -70:
        sig = "🔥 GÜÇLÜ SHORT"
    elif score <= -30:
        sig = "🔴 SHORT"
    else:
        sig = "⚪ BEKLE"

    return sig, score, reasons, float(l15["close"])

# =========================
# RUN
# =========================
status_box = st.empty()

symbols_df, sym_err = get_symbols_top_usdt(universe_n=universe_n)
if symbols_df is None:
    st.error(f"Binance verisi alınamadı. Biraz sonra tekrar deneyin.\n\nDetay: {sym_err}")
    st.stop()

results = []
errors = 0

# çok agresif istek atmayalım
symbols = symbols_df["symbol"].tolist()

progress = st.progress(0, text="Tarama başlıyor...")

for i, sym in enumerate(symbols):
    try:
        d15, e1 = get_klines(sym, "15m", 220)
        d4h, e2 = get_klines(sym, "4h", 220)

        if d15 is None or d4h is None:
            errors += 1
            continue

        sig, score, reasons, price = score_signal(d15, d4h)

        if only_strong and ("GÜÇLÜ" not in sig):
            pass
        else:
            results.append({
                "Coin": sym.replace("USDT", "/USDT"),
                "Sinyal": sig,
                "Skor": score,
                "Fiyat": price,
                "Gerekçe": " | ".join(reasons)
            })

        # hafif jitter (rate limitten kaçın)
        if i % 6 == 0:
            _sleep_jitter(0.12, 0.28)

    except Exception:
        errors += 1

    progress.progress((i + 1) / len(symbols), text=f"Taranıyor: {i+1}/{len(symbols)}")

progress.empty()

if not results:
    st.warning("Sinyal üretildi ama filtre yüzünden tablo boş olabilir. (Sadece güçlü açıksa kapatıp dene)")
    st.stop()

df = pd.DataFrame(results)

# Top-N: hem long hem short en güçlüleri göstermek için abs skora göre sıralayalım
df["AbsSkor"] = df["Skor"].abs()
df = df.sort_values(["AbsSkor", "Skor"], ascending=[False, False]).drop(columns=["AbsSkor"]).head(int(TOP_N))

# Sade tablo
st.dataframe(df, use_container_width=True, hide_index=True)

# alt bilgi
status_box.info(
    f"Son güncelleme: {datetime.now().strftime('%H:%M:%S')}  |  "
    f"Evren: {len(symbols)}  |  Hata/Atlanan: {errors}  |  "
    f"Yenileme: {REFRESH_SEC}s"
)

# Otomatik yenile
time.sleep(int(REFRESH_SEC))
st.rerun()
