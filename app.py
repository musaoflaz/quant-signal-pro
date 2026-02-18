# requirements.txt (install these):
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
# FIXED SETTINGS (NO SIDEBAR)
# =============================
IST_TZ = ZoneInfo("Europe/Istanbul")

TIMEFRAME = "15m"
CANDLE_LIMIT = 200
AUTO_REFRESH_SEC = 240

# Universe & liquidity
MIN_QV_USDT = 200_000       # minimum 24h quoteVolume (USDT)
MAX_SCAN_UNIVERSE = 450     # scan top-N by liquidity after filtering

# Indicators
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIOD = 20

# Strong gate (RAW)
RAW_STRONG_LONG = 90
RAW_STRONG_SHORT = 10

# Fallback table (if no STRONG)
FALLBACK_LONG = 10
FALLBACK_SHORT = 10

# Output
TOP_SNIPER = 3  # if STRONG exists, show only top 3 strongest

# Exclude stable/fiat-like bases to reduce noise
EXCLUDED_BASE = {
    "USDT", "USDC", "TUSD", "USDE", "DAI", "FDUSD", "USDP", "BUSD",
    "EUR", "TRY", "GBP", "JPY", "CHF", "AUD", "CAD",
    "PAXG",
}


# =============================
# Exchange
# =============================
def make_exchange() -> ccxt.kucoin:
    return ccxt.kucoin({"enableRateLimit": True, "timeout": 20000})


@st.cache_data(show_spinner=False, ttl=600)
def load_usdt_spot_symbols() -> list[str]:
    ex = make_exchange()
    markets = ex.load_markets()
    syms: list[str] = []

    for sym, m in markets.items():
        if not m:
            continue
        if not m.get("active", True):
            continue
        if not m.get("spot", False):
            continue
        if m.get("quote") != "USDT":
            continue

        base = (m.get("base") or "").upper().strip()
        if base in EXCLUDED_BASE:
            continue

        syms.append(sym)

    return sorted(set(syms))


def safe_fetch_tickers(ex: ccxt.Exchange, symbols: list[str]) -> dict:
    try:
        return ex.fetch_tickers(symbols)
    except Exception:
        try:
            all_t = ex.fetch_tickers()
            return {s: all_t.get(s) for s in symbols if s in all_t}
        except Exception:
            return {}


def qv_usdt(t: dict) -> float:
    if not t or not isinstance(t, dict):
        return 0.0
    qv = t.get("quoteVolume")
    if qv is not None:
        try:
            return float(qv)
        except Exception:
            pass
    bv = t.get("baseVolume")
    last = t.get("last")
    try:
        if bv is not None and last is not None:
            return float(bv) * float(last)
    except Exception:
        pass
    return 0.0


def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


# =============================
# Indicators (pure pandas/numpy)
# =============================
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def bollinger(series: pd.Series, period: int, n_std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + (n_std * std)
    lower = mid - (n_std * std)
    return mid, upper, lower


def rsi_wilder(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


# =============================
# RAW score (0/100 extremes were too common before)
# We keep your original 90+/10- gate, but compute RAW with a
# more continuous scale so it doesn't always end up 0 or 100.
# =============================
def compute_raw_score(close: float, sma20: float, rsi14: float, bb_low: float, bb_up: float) -> int:
    # base
    raw = 50.0

    # Trend component: distance vs SMA (smooth)
    if sma20 > 0:
        dist = (close - sma20) / sma20  # ~ -0.05..+0.05 mostly
        raw += np.clip(dist * 400.0, -20.0, 20.0)  # max +/-20

    # RSI component: oversold/overbought smooth
    # below 35 -> positive; above 65 -> negative
    if rsi14 < 50:
        raw += np.clip((50 - rsi14) * 1.6, 0.0, 40.0)
    else:
        raw -= np.clip((rsi14 - 50) * 1.6, 0.0, 40.0)

    # Bollinger component: how far outside bands (smooth)
    band_width = max(bb_up - bb_low, 1e-12)
    if close < bb_low:
        raw += np.clip(((bb_low - close) / band_width) * 80.0, 0.0, 40.0)
    elif close > bb_up:
        raw -= np.clip(((close - bb_up) / band_width) * 80.0, 0.0, 40.0)
    else:
        # inside bands, small push toward mean reversion
        mid = (bb_low + bb_up) / 2.0
        raw += np.clip(((mid - close) / band_width) * 20.0, -10.0, 10.0)

    raw = float(np.clip(raw, 0.0, 100.0))
    return int(round(raw))


def direction_from_raw(raw: int) -> str:
    # This is just table direction
    if raw >= 50:
        return "LONG"
    return "SHORT"


def label_from_raw(raw: int) -> str:
    if raw >= RAW_STRONG_LONG:
        return "🔥 STRONG LONG"
    if raw <= RAW_STRONG_SHORT:
        return "💀 STRONG SHORT"
    return "⏳ WATCH"


# =============================
# UI helpers (fix deprecation spam + readability)
# =============================
def dataframe_safe(obj, height: int = 650):
    """
    Streamlit warning fix:
    New versions prefer width="stretch".
    Old versions still accept use_container_width=True.
    """
    try:
        st.dataframe(obj, width="stretch", height=height)
    except TypeError:
        st.dataframe(obj, use_container_width=True, height=height)


def style_table(df: pd.DataFrame):
    def dir_style(v):
        if v == "LONG":
            return "background-color:#064e3b;color:#ffffff;font-weight:700;"
        if v == "SHORT":
            return "background-color:#7f1d1d;color:#ffffff;font-weight:700;"
        return ""

    def raw_style(v):
        try:
            v = float(v)
        except Exception:
            return ""
        if v >= RAW_STRONG_LONG:
            return "background-color:#006400;color:#ffffff;font-weight:800;"
        if v <= RAW_STRONG_SHORT:
            return "background-color:#8B0000;color:#ffffff;font-weight:800;"
        return ""

    fmt = {
        "FİYAT": "{:.4f}",
        "RAW": "{:.0f}",
        "QV_24H": "{:,.0f}",
    }

    return (
        df.style.format(fmt)
        .applymap(dir_style, subset=["YÖN"])
        .applymap(raw_style, subset=["RAW"])
        .set_properties(**{"border-color": "#1f2a37"})
    )


# =============================
# SCAN
# =============================
def run_scan() -> pd.DataFrame:
    ex = make_exchange()

    syms = load_usdt_spot_symbols()
    tickers = safe_fetch_tickers(ex, syms)

    ranked = []
    for s in syms:
        t = tickers.get(s) or {}
        qv = qv_usdt(t)
        if qv >= MIN_QV_USDT:
            ranked.append((s, qv))

    ranked.sort(key=lambda x: x[1], reverse=True)
    universe = [s for s, _ in ranked[:MAX_SCAN_UNIVERSE]]

    rows = []
    total = len(universe)

    prog = st.progress(0, text="Tarama başlıyor…")
    status = st.empty()

    for i, symbol in enumerate(universe, start=1):
        prog.progress(int((i - 1) / max(total, 1) * 100), text=f"{symbol} ({i}/{total}) inceleniyor…")

        try:
            ohlcv = safe_fetch_ohlcv(ex, symbol, TIMEFRAME, CANDLE_LIMIT)
            if not ohlcv or len(ohlcv) < max(RSI_PERIOD, BB_PERIOD, SMA_PERIOD) + 5:
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            close = df["close"].astype(float)

            sma20_s = sma(close, SMA_PERIOD)
            _, bb_up_s, bb_low_s = bollinger(close, BB_PERIOD, BB_STD)
            rsi_s = rsi_wilder(close, RSI_PERIOD)

            last_close = float(close.iloc[-1])
            last_sma20 = float(sma20_s.iloc[-1])
            last_rsi = float(rsi_s.iloc[-1])
            last_low = float(bb_low_s.iloc[-1])
            last_up = float(bb_up_s.iloc[-1])

            if any(np.isnan([last_close, last_sma20, last_rsi, last_low, last_up])):
                continue

            raw = compute_raw_score(last_close, last_sma20, last_rsi, last_low, last_up)
            yon = direction_from_raw(raw)
            label = label_from_raw(raw)

            t = tickers.get(symbol) or {}
            qv = qv_usdt(t)

            rows.append(
                {
                    "COIN": symbol.replace("/USDT", ""),
                    "YÖN": yon,
                    "SKOR": raw,     # keep SKOR=RAW for now
                    "FİYAT": last_close,
                    "RAW": raw,
                    "QV_24H": qv,
                    "ETİKET": label,
                }
            )

        except (ccxt.RequestTimeout, ccxt.NetworkError):
            status.warning(f"Network/Timeout: {symbol}")
        except ccxt.ExchangeError:
            status.info(f"Exchange error: {symbol}")
        except Exception:
            # silent skip for robustness
            pass

        time.sleep(0.02)

    prog.progress(100, text="Tarama bitti.")
    status.empty()

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Sorting: strongest longs near 100, strongest shorts near 0
    out["LONG_KEY"] = np.where(out["YÖN"] == "LONG", out["RAW"], -1)
    out["SHORT_KEY"] = np.where(out["YÖN"] == "SHORT", 100 - out["RAW"], -1)

    # We'll keep a single "priority" sort:
    # 1) STRONG first
    out["IS_STRONG"] = ((out["RAW"] >= RAW_STRONG_LONG) | (out["RAW"] <= RAW_STRONG_SHORT)).astype(int)

    # 2) Then closeness to extremes (longs high, shorts low)
    out["EXTREME_DIST"] = np.where(out["YÖN"] == "LONG", 100 - out["RAW"], out["RAW"])

    out = out.sort_values(["IS_STRONG", "EXTREME_DIST"], ascending=[False, True])

    return out.drop(columns=["LONG_KEY", "SHORT_KEY", "IS_STRONG", "EXTREME_DIST"]).reset_index(drop=True)


# =============================
# PAGE
# =============================
st.set_page_config(page_title="KuCoin PRO Sniper", layout="wide")

# Strong CSS to fix “white page / unreadable text”
st.markdown(
    """
<style>
/* Force dark background everywhere */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"]{
  background: #0b0f14 !important;
}
* { color: #e6edf3 !important; }

/* Nice containers */
.block-container { padding-top: 1.1rem; }

/* Make status boxes readable */
div[data-testid="stAlert"] * { color: #0b0f14 !important; }
div[data-testid="stAlert"] { filter: none !important; opacity: 1 !important; }

/* Buttons if any */
.stButton>button {
  border: 1px solid #2d3b4d !important;
  background: #111827 !important;
  color: #e6edf3 !important;
  border-radius: 10px !important;
  padding: 0.55rem 0.9rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Auto refresh
try:
    st.autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
except Exception:
    try:
        st.experimental_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
    except Exception:
        pass

now_ist = datetime.now(IST_TZ)

st.title("KuCoin PRO Sniper — Direkt Tablo (Sade)")
st.caption(f"TF={TIMEFRAME} • Strong: RAW≥{RAW_STRONG_LONG} LONG / RAW≤{RAW_STRONG_SHORT} SHORT • Auto refresh: {AUTO_REFRESH_SEC}s")
st.markdown(
    f"""
<div style="display:flex;justify-content:flex-end;">
  <div style="padding:8px 12px;border:1px solid #1f2a37;border-radius:10px;background:#0f172a;">
    <div style="font-size:12px;">Istanbul Time</div>
    <div style="font-size:18px;font-weight:800;">{now_ist.strftime("%Y-%m-%d %H:%M:%S")}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Loading indicator (so you don't stare at blank)
with st.spinner("⏳ Tarama yapılıyor… KuCoin verileri çekiliyor, lütfen bekle."):
    df_all = run_scan()

st.write("")
st.subheader("🎯 SNIPER TABLO")

if df_all is None or df_all.empty:
    st.warning("Aday yok (network/KuCoin veya filtre çok sert olabilir). Bir sonraki yenilemede tekrar dener.")
    st.stop()

# Split strong
strong = df_all[(df_all["RAW"] >= RAW_STRONG_LONG) | (df_all["RAW"] <= RAW_STRONG_SHORT)].copy()

if not strong.empty:
    # show top sniper only (best 3 extremes)
    # longs strongest: highest RAW; shorts strongest: lowest RAW
    strong["PRIO"] = np.where(strong["YÖN"] == "LONG", 100 - strong["RAW"], strong["RAW"])
    strong = strong.sort_values("PRIO", ascending=True).drop(columns=["PRIO"]).head(TOP_SNIPER)

    st.success(f"✅ STRONG bulundu. En güçlü {min(TOP_SNIPER, len(strong))} sinyal gösteriliyor.")
    show = strong.loc[:, ["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H"]].copy()
    dataframe_safe(style_table(show), height=520)
else:
    st.info(f"Şu an STRONG yok. En yakın adaylardan {FALLBACK_LONG} LONG + {FALLBACK_SHORT} SHORT gösteriyorum.")

    longs = df_all[df_all["YÖN"] == "LONG"].sort_values("RAW", ascending=False).head(FALLBACK_LONG)
    shorts = df_all[df_all["YÖN"] == "SHORT"].sort_values("RAW", ascending=True).head(FALLBACK_SHORT)

    show = pd.concat([longs, shorts], ignore_index=True)

    # Sort for nice view: longs first then shorts, both by closeness to strong
    show["VIEWKEY"] = np.where(show["YÖN"] == "LONG", 100 - show["RAW"], show["RAW"])
    show = show.sort_values(["YÖN", "VIEWKEY"], ascending=[True, True]).drop(columns=["VIEWKEY"])

    show = show.loc[:, ["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H"]].copy()
    dataframe_safe(style_table(show), height=650)
