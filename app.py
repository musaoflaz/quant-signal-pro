# requirements.txt:
# streamlit
# pandas
# numpy
# ccxt

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import ccxt


# =============================
# SABİT AYARLAR (MENÜ YOK)
# =============================
IST = ZoneInfo("Europe/Istanbul")

TF_MAIN = "15m"          # ana sinyal timeframe
TF_TREND = "1h"          # trend onayı timeframe (Kapı-2)
CANDLE_LIMIT_MAIN = 220
CANDLE_LIMIT_TREND = 160

AUTO_REFRESH_SEC = 240
TABLE_SIZE = 20

# STRONG eşiği
STRONG_MIN_SCORE = 90

# Skor adımı (RAW = ham, SKOR = yuvarlanmış)
SCORE_STEP = 5

# Evren çok büyük -> stabilite için soft likidite filtre + cap
MIN_QUOTE_VOL_24H = 10_000
MAX_SCAN_SYMBOLS = 450

# =============================
# SEVİYE 2 / 6 KAPI (STRONG GATE)
# =============================
# Kapı-1: Likidite + Spread
GATE_MIN_QV_24H = 50_000          # USDT (düşük hacim tuzaklarını azaltır)
GATE_MAX_SPREAD_PCT = 0.35        # % (ticker bid/ask varsa)

# Kapı-2: Trend onayı (15m + 1h)
# LONG: EMA20 > EMA50 ve fiyat EMA20 üstü (iki TF)
# SHORT: EMA20 < EMA50 ve fiyat EMA20 altı (iki TF)

# Kapı-3: Momentum onayı (RSI bölge + dönüş)
GATE_RSI_LONG_MAX = 35.0
GATE_RSI_SHORT_MIN = 65.0

# Kapı-4: Volatilite filtresi (ATR%)
GATE_ATR_PCT_MIN = 0.40           # çok sıkışık olmasın
GATE_ATR_PCT_MAX = 6.00           # aşırı çılgın olmasın

# Kapı-5: Spike/Wick filtresi
GATE_MAX_WICK_RATIO = 0.60        # iğne ağırlıklı mumları kes
GATE_MAX_RANGE_ATR = 2.8          # son mum range / ATR

# Kapı-6: Hacim onayı (Volume spike)
GATE_MIN_VOL_RATIO = 1.30         # son hacim / hacim SMA20

# Ağırlıklar (mevcut skor mantığı: LONG/SHORT ayrı, max alınır)
W_RSI = 22
W_BB = 22
W_TREND = 18
W_MACD = 14
W_ADX = 10
W_ATR = 8
W_VOL = 6


# =============================
# KUCOIN / CCXT
# =============================
def make_exchange() -> ccxt.kucoin:
    return ccxt.kucoin({"enableRateLimit": True, "timeout": 20000})


@st.cache_data(show_spinner=False, ttl=600)
def load_usdt_spot_symbols() -> list[str]:
    ex = make_exchange()
    markets = ex.load_markets()
    out: list[str] = []
    for sym, m in markets.items():
        if not m:
            continue
        if not m.get("active", True):
            continue
        if not m.get("spot", False):
            continue
        if m.get("quote") != "USDT":
            continue
        out.append(sym)
    return sorted(set(out))


def safe_fetch_tickers(ex: ccxt.Exchange) -> dict:
    try:
        return ex.fetch_tickers()
    except Exception:
        return {}


def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def qv_24h(t: dict) -> float:
    if not t or not isinstance(t, dict):
        return 0.0
    qv = t.get("quoteVolume")
    if qv is not None:
        try:
            return float(qv)
        except Exception:
            return 0.0
    # fallback
    bv = t.get("baseVolume")
    last = t.get("last")
    try:
        if bv is not None and last is not None:
            return float(bv) * float(last)
    except Exception:
        pass
    return 0.0


def spread_pct_from_ticker(t: dict) -> float | None:
    """
    bid/ask varsa spread% hesaplar. Yoksa None.
    """
    if not t or not isinstance(t, dict):
        return None
    bid = t.get("bid")
    ask = t.get("ask")
    try:
        if bid is None or ask is None:
            return None
        bid = float(bid)
        ask = float(ask)
        if bid <= 0 or ask <= 0:
            return None
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return None
        return ((ask - bid) / mid) * 100.0
    except Exception:
        return None


# =============================
# İNDİKATÖRLER (PURE pandas/numpy)
# =============================
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    up = mid + k * std
    low = mid - k * std
    return mid, up, low


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ef = ema(close, fast)
    es = ema(close, slow)
    line = ef - es
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = true_range(high, low, close)
    atr_n = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_n

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().fillna(0.0)


# =============================
# SKORLAMA (mevcut mantık)
# =============================
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def quantize_score(x: float, step: int = 5) -> int:
    x = clamp(x, 0, 100)
    q = int(round(x / step) * step)
    return int(clamp(q, 0, 100))


def score_main(df: pd.DataFrame) -> dict:
    """
    Ana skoru ve kapıların ihtiyaç duyduğu metrikleri üretir.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    vol = df["volume"].astype(float)

    if len(close) < 80:
        return {
            "ok": False,
        }

    rsi_s = rsi_wilder(close, 14)
    rsi = float(rsi_s.iloc[-1])
    rsi_prev = float(rsi_s.iloc[-2])

    _, up, low_bb = bollinger(close, 20, 2.0)
    sma20 = sma(close, 20)

    _, _, macd_hist = macd(close)
    hist = float(macd_hist.iloc[-1])

    adx_v = float(adx(high, low, close, 14).iloc[-1])
    atr_s = atr(high, low, close, 14)
    atr_v = float(atr_s.iloc[-1])

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])

    last_sma = float(sma20.iloc[-1])
    prev_sma = float(sma20.iloc[-2])

    bb_up = float(up.iloc[-1])
    bb_low = float(low_bb.iloc[-1])
    bb_range = max(1e-9, (bb_up - bb_low))
    bb_pos = (last - bb_low) / bb_range  # 0..1

    # ATR%
    atr_pct = (atr_v / last) * 100.0 if last > 0 else 0.0

    # Volume ratio
    vol_sma = float(sma(vol, 20).iloc[-1]) if len(vol) >= 20 else float(vol.mean())
    vol_ratio = (float(vol.iloc[-1]) / vol_sma) if vol_sma > 0 else 1.0

    # Trend slope
    sma_slope = last_sma - prev_sma

    # Wick ratio (son mum)
    o = float(open_.iloc[-1])
    h = float(high.iloc[-1])
    l = float(low.iloc[-1])
    c = float(close.iloc[-1])
    body = abs(c - o)
    rng = max(1e-9, (h - l))
    wick_ratio = clamp((rng - body) / rng, 0.0, 1.0)

    # Range / ATR
    range_atr = (rng / (atr_v + 1e-9)) if atr_v > 0 else 0.0

    # ---------- LONG / SHORT ayrı ham puan ----------
    long_raw = 0.0
    short_raw = 0.0

    # RSI
    long_raw += W_RSI * clamp((50.0 - rsi) / 25.0, 0.0, 1.0)
    short_raw += W_RSI * clamp((rsi - 50.0) / 25.0, 0.0, 1.0)

    # Bollinger
    long_raw += W_BB * clamp((0.35 - bb_pos) / 0.35, 0.0, 1.0)
    short_raw += W_BB * clamp((bb_pos - 0.65) / 0.35, 0.0, 1.0)

    # Trend (SMA20 + slope)
    if last > last_sma and sma_slope > 0:
        long_raw += W_TREND
    elif last < last_sma and sma_slope < 0:
        short_raw += W_TREND
    else:
        if last > last_sma:
            long_raw += W_TREND * 0.35
        elif last < last_sma:
            short_raw += W_TREND * 0.35

    # MACD hist
    hist_norm = clamp(abs(hist) / (abs(last) * 0.002 + 1e-9), 0.0, 1.0)
    if hist > 0:
        long_raw += W_MACD * hist_norm
    elif hist < 0:
        short_raw += W_MACD * hist_norm

    # ADX
    adx_mult = 0.65 + 0.35 * clamp((adx_v - 18.0) / 12.0, 0.0, 1.0)
    adx_bonus = W_ADX * clamp((adx_v - 18.0) / 18.0, 0.0, 1.0)
    long_raw += adx_bonus * (1.0 if last > last_sma else 0.4)
    short_raw += adx_bonus * (1.0 if last < last_sma else 0.4)

    # ATR% penalty
    if atr_pct <= 4.0:
        atr_mult = 1.0
    elif atr_pct >= 8.0:
        atr_mult = 0.75
    else:
        atr_mult = 1.0 - (atr_pct - 4.0) * (0.25 / 4.0)

    atr_bonus = W_ATR * clamp((4.0 - atr_pct) / 4.0, 0.0, 1.0)
    long_raw += atr_bonus * 0.6
    short_raw += atr_bonus * 0.6

    # Volume bonus
    vol_boost = W_VOL * clamp((vol_ratio - 1.2) / 1.0, 0.0, 1.0)
    if last > last_sma:
        long_raw += vol_boost
    elif last < last_sma:
        short_raw += vol_boost

    long_raw *= adx_mult * atr_mult
    short_raw *= adx_mult * atr_mult

    weight_sum = float(W_RSI + W_BB + W_TREND + W_MACD + W_ADX + W_ATR + W_VOL)
    long_score = (long_raw / weight_sum) * 100.0
    short_score = (short_raw / weight_sum) * 100.0

    if long_score >= short_score:
        direction = "LONG"
        raw_best = int(round(clamp(long_score, 0, 100)))
        score_q = quantize_score(long_score, SCORE_STEP)
    else:
        direction = "SHORT"
        raw_best = int(round(clamp(short_score, 0, 100)))
        score_q = quantize_score(short_score, SCORE_STEP)

    return {
        "ok": True,
        "direction": direction,
        "score": int(score_q),
        "raw": int(raw_best),
        "last": float(last),
        "rsi": float(rsi),
        "rsi_prev": float(rsi_prev),
        "atr_pct": float(atr_pct),
        "vol_ratio": float(vol_ratio),
        "wick_ratio": float(wick_ratio),
        "range_atr": float(range_atr),
        "sma20": float(last_sma),
    }


# =============================
# 6 KAPI / STRONG GATE
# =============================
def trend_ok_mtf(ex: ccxt.Exchange, symbol: str, direction: str) -> bool:
    """
    Kapı-2: 15m + 1h trend onayı.
    15m trend: zaten main df'de SMA20 referanslı yaklaşıyoruz.
    1h trend: EMA20/EMA50 + fiyat EMA20.
    """
    try:
        ohlcv = safe_fetch_ohlcv(ex, symbol, TF_TREND, CANDLE_LIMIT_TREND)
        if not ohlcv or len(ohlcv) < 60:
            return False
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        close = df["close"].astype(float)

        e20 = ema(close, 20)
        e50 = ema(close, 50)

        last = float(close.iloc[-1])
        le20 = float(e20.iloc[-1])
        le50 = float(e50.iloc[-1])

        if any(np.isnan([last, le20, le50])):
            return False

        if direction == "LONG":
            return (le20 > le50) and (last > le20)
        else:
            return (le20 < le50) and (last < le20)
    except Exception:
        return False


def gates_pass(
    ex: ccxt.Exchange,
    symbol: str,
    direction: str,
    main_metrics: dict,
    qv24: float,
    spread_pct: float | None,
) -> tuple[bool, list[bool]]:
    """
    Returns: (all_pass, [k1..k6])
    """
    # Kapı-1: Likidite + Spread (spread yoksa sadece qv)
    k1_qv = qv24 >= GATE_MIN_QV_24H
    if spread_pct is None:
        k1 = k1_qv
    else:
        k1 = k1_qv and (spread_pct <= GATE_MAX_SPREAD_PCT)

    # Kapı-2: MTF Trend
    k2 = trend_ok_mtf(ex, symbol, direction)

    # Kapı-3: Momentum (RSI + dönüş)
    rsi = float(main_metrics["rsi"])
    rsi_prev = float(main_metrics["rsi_prev"])
    if direction == "LONG":
        k3 = (rsi <= GATE_RSI_LONG_MAX) and (rsi > rsi_prev)
    else:
        k3 = (rsi >= GATE_RSI_SHORT_MIN) and (rsi < rsi_prev)

    # Kapı-4: ATR% band
    atr_pct = float(main_metrics["atr_pct"])
    k4 = (atr_pct >= GATE_ATR_PCT_MIN) and (atr_pct <= GATE_ATR_PCT_MAX)

    # Kapı-5: Wick / Spike
    wick_ratio = float(main_metrics["wick_ratio"])
    range_atr = float(main_metrics["range_atr"])
    k5 = (wick_ratio <= GATE_MAX_WICK_RATIO) and (range_atr <= GATE_MAX_RANGE_ATR)

    # Kapı-6: Volume confirmation
    vol_ratio = float(main_metrics["vol_ratio"])
    k6 = vol_ratio >= GATE_MIN_VOL_RATIO

    checks = [k1, k2, k3, k4, k5, k6]
    return all(checks), checks


# =============================
# UI (Koyu tema + okunurluk fix)
# =============================
st.set_page_config(page_title="KuCoin PRO Sniper — Level2 (6 Gates)", layout="wide")

st.markdown(
    """
<style>
:root { color-scheme: dark !important; }

html, body, [data-testid="stAppViewContainer"]{
  background: #0b0f14 !important;
  color: #e6edf3 !important;
}

* { opacity: 1 !important; }

h1,h2,h3,h4,h5,h6,p,span,div,label{
  color: #e6edf3 !important;
}

[data-testid="stHeader"]{ background: rgba(0,0,0,0) !important; }

[data-testid="stDataFrame"]{
  background: #0b0f14 !important;
  border: 1px solid #1f2a37 !important;
}

thead tr th{
  background: #0f172a !important;
  color: #e6edf3 !important;
  border-bottom: 1px solid #1f2a37 !important;
}
tbody tr td{
  border-bottom: 1px solid #111827 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

def try_autorefresh(interval_ms: int, key: str):
    try:
        return st.autorefresh(interval=interval_ms, key=key)
    except Exception:
        try:
            return st.experimental_autorefresh(interval=interval_ms, key=key)
        except Exception:
            return None

try_autorefresh(interval_ms=int(AUTO_REFRESH_SEC * 1000), key="auto_refresh")


now_ist = datetime.now(IST)
st.title("KuCoin PRO Sniper — Seviye 2 (6 Kapı) • Auto")
st.caption(
    f"TF={TF_MAIN} • STRONG: SKOR≥{STRONG_MIN_SCORE} + 6 Kapı • Auto: {AUTO_REFRESH_SEC}s • SKOR adımı: {SCORE_STEP}"
)
st.markdown(
    f"""
<div style="text-align:right; margin-top:-40px;">
  <div style="font-size:12px; opacity:0.85;">Istanbul Time</div>
  <div style="font-size:18px; font-weight:800;">{now_ist.strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")


# =============================
# TARAMA (başarılı mantık + kapılar)
# =============================
ex = make_exchange()

errors: list[str] = []

with st.spinner("⏳ KuCoin USDT spot evreni taranıyor..."):
    syms = load_usdt_spot_symbols()
    tickers = safe_fetch_tickers(ex)

# Likidite filtre + cap (stabilite için)
ranked = []
for s in syms:
    t = tickers.get(s)
    qv = qv_24h(t)
    if qv >= MIN_QUOTE_VOL_24H:
        ranked.append((s, qv))
ranked.sort(key=lambda x: x[1], reverse=True)
scan_list = [s for s, _ in ranked[:MAX_SCAN_SYMBOLS]]

st.info(
    f"Evren (USDT spot): {len(syms)} • Likidite filtresi sonrası: {len(ranked)} • Tarama: {len(scan_list)}",
    icon="🧠",
)

progress = st.progress(0)
status = st.empty()

rows: list[dict] = []
total = max(1, len(scan_list))

# Kapı-2 (1h) pahalı olduğu için:
# önce ana skorları çıkarırız, sonra STRONG adayı olabilecekleri kapıya sokarız.
candidates_for_gates: list[tuple[str, dict, float, float | None]] = []

for i, symbol in enumerate(scan_list, start=1):
    if i == 1:
        status.write("🔎 Mum verileri çekiliyor, skor hesaplanıyor...")

    progress.progress(int((i / total) * 100))

    try:
        ohlcv = safe_fetch_ohlcv(ex, symbol, TF_MAIN, CANDLE_LIMIT_MAIN)
        if not ohlcv or len(ohlcv) < 80:
            continue

        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        m = score_main(df)
        if not m.get("ok"):
            continue

        t = tickers.get(symbol, {}) or {}
        qv = qv_24h(t)
        sp = spread_pct_from_ticker(t)

        # fiyat
        last = t.get("last", None)
        if last is None:
            last = float(df["close"].iloc[-1])
        else:
            try:
                last = float(last)
            except Exception:
                last = float(df["close"].iloc[-1])

        score_q = int(m["score"])
        raw = int(m["raw"])
        direction = str(m["direction"])

        # Gate aday havuzu:
        # 1) score>=85 olanları kapılara sok (90+ için ön-eleme)
        # 2) ayrıca tablo dolsun diye uç adayları da sokmayacağız; sadece STRONG kontrol için yeter.
        if score_q >= 85:
            candidates_for_gates.append((symbol, m, qv, sp))

        rows.append(
            {
                "YÖN": direction,
                "COIN": symbol.replace("/USDT", ""),
                "SKOR": score_q,
                "RAW": raw,
                "FİYAT": float(last),
                "RSI": float(m["rsi"]),
                "ATR%": float(m["atr_pct"]),
                "VOLx": float(m["vol_ratio"]),
                "QV_24H": float(qv),
                "SPREAD%": float(sp) if sp is not None else np.nan,
                "SYMBOL": symbol,
                "STRONG": False,  # sonra dolduracağız
                "GATES": "—",      # sonra dolduracağız
            }
        )

    except (ccxt.RequestTimeout, ccxt.NetworkError):
        continue
    except Exception as e:
        # çok spam olmasın diye kısalt
        errors.append(f"{symbol}: {type(e).__name__}")
        continue

    time.sleep(0.02)

progress.empty()
status.empty()

df_all = pd.DataFrame(rows)
if df_all.empty:
    st.error("Sonuç yok. (KuCoin ağ / rate-limit olabilir) Bir sonraki auto refresh’i bekle.")
    if errors:
        with st.expander("Hata detayları"):
            for e in errors[:50]:
                st.write("-", e)
    st.stop()

# 6 kapı kontrolü (seviye2) — sadece adaylar için
strong_count = 0
gate_pass_map: dict[str, tuple[bool, str]] = {}

for symbol, m, qv, sp in candidates_for_gates:
    try:
        direction = str(m["direction"])
        # STRONG eşiği: 90+
        if int(m["score"]) < STRONG_MIN_SCORE:
            continue

        ok, checks = gates_pass(ex, symbol, direction, m, qv, sp)
        # 6 kapı string
        gate_str = "".join(["✅" if c else "❌" for c in checks])  # k1..k6
        gate_pass_map[symbol] =
