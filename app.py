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
# CONFIG (menüsüz, sade)
# =============================
IST_TZ = ZoneInfo("Europe/Istanbul")

TF = "15m"
CANDLE_LIMIT = 200
AUTO_REFRESH_SEC = 240

TOP_ROWS = 20
BALANCED_FALLBACK = True  # Strong yoksa 10 LONG + 10 SHORT denge

RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIOD = 20

STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10

# Çok ölü marketler falselar üretiyor; çok düşük tuttuk (evren geniş kalsın)
MIN_QV_USDT = 10_000.0

EXCHANGE_TIMEOUT_MS = 20000


# =============================
# INDICATORS
# =============================
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def bollinger_bands(series: pd.Series, period: int, n_std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
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


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    m_fast = ema(series, fast)
    m_slow = ema(series, slow)
    macd = m_fast - m_slow
    sig = ema(macd, signal)
    return macd - sig


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


# =============================
# EXCHANGE
# =============================
def make_exchange() -> ccxt.kucoin:
    return ccxt.kucoin({"enableRateLimit": True, "timeout": EXCHANGE_TIMEOUT_MS})


@st.cache_data(show_spinner=False, ttl=3600)
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


def safe_fetch_tickers(ex: ccxt.Exchange, symbols: list[str]) -> dict:
    try:
        return ex.fetch_tickers(symbols)
    except Exception:
        try:
            all_t = ex.fetch_tickers()
            return {s: all_t.get(s) for s in symbols if s in all_t}
        except Exception:
            return {}


def quote_volume_usdt(t: dict | None) -> float:
    if not t or not isinstance(t, dict):
        return 0.0
    qv = t.get("quoteVolume")
    if qv is not None:
        try:
            return float(qv)
        except Exception:
            return 0.0
    bv = t.get("baseVolume")
    last = t.get("last")
    try:
        if bv is not None and last is not None:
            return float(bv) * float(last)
    except Exception:
        return 0.0
    return 0.0


def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


# =============================
# SCORING (kademeli, 1'er 1'er)
# =============================
def score_raw(close: float, sma20: float, rsi14: float, bb_low: float, bb_up: float, macd_h: float) -> int:
    base = 50.0

    # Trend (±20)
    if sma20 <= 0:
        trend_pts = 0.0
    else:
        rel = (close - sma20) / sma20
        trend_pts = clamp(rel / 0.03, -1.0, 1.0) * 20.0  # %3 cap

    # RSI (±30)
    if rsi14 < 35:
        rsi_pts = clamp((35.0 - rsi14) / 25.0, 0.0, 1.0) * 30.0
    elif rsi14 > 65:
        rsi_pts = -clamp((rsi14 - 65.0) / 25.0, 0.0, 1.0) * 30.0
    else:
        rsi_pts = 0.0

    # Bollinger (±30)
    band_w = (bb_up - bb_low)
    if band_w <= 0:
        bb_pts = 0.0
    else:
        pos = (close - bb_low) / band_w  # 0=lower,1=upper
        if pos <= 0.5:
            bb_pts = clamp((0.5 - pos) / 0.5, 0.0, 1.0) * 30.0
        else:
            bb_pts = -clamp((pos - 0.5) / 0.5, 0.0, 1.0) * 30.0

    # MACD teyidi (±20)
    macd_mag = clamp(abs(macd_h) / max(abs(close) * 0.002, 1e-9), 0.0, 1.0)
    if trend_pts >= 0:
        macd_pts = (20.0 * macd_mag) if macd_h > 0 else (-10.0 * macd_mag)
    else:
        macd_pts = (20.0 * macd_mag) if macd_h < 0 else (-10.0 * macd_mag)

    raw = base + trend_pts + rsi_pts + bb_pts + macd_pts
    raw = clamp(raw, 0.0, 100.0)
    return int(round(raw))


def direction_from_raw(raw: int) -> str:
    return "LONG" if raw >= 50 else "SHORT"


# =============================
# TABLE BUILD (STRONG + fallback dengeli)
# =============================
def build_sniper_table(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, dict]:
    """
    Öncelik:
    1) STRONG LONG (RAW>=90) + STRONG SHORT (RAW<=10) -> önce
    2) Boş kalırsa:
       - BALANCED_FALLBACK=True ise 10 LONG + 10 SHORT "en yakın" aday (90'a ve 10'a yakın)
       - değilse RAW'a göre top doldur
    """
    meta = {"strong_long": 0, "strong_short": 0, "mode": ""}

    if df.empty:
        meta["mode"] = "empty"
        return df, meta

    strong_long = df[df["RAW"] >= STRONG_LONG_MIN].copy().sort_values(["RAW", "QV_24H"], ascending=[False, False])
    strong_short = df[df["RAW"] <= STRONG_SHORT_MAX].copy().sort_values(["RAW", "QV_24H"], ascending=[True, False])

    meta["strong_long"] = int(len(strong_long))
    meta["strong_short"] = int(len(strong_short))

    strong = pd.concat([strong_long, strong_short], ignore_index=True)
    strong = strong.drop_duplicates(subset=["PAIR"], keep="first")

    if len(strong) >= top_n:
        meta["mode"] = "strong_only"
        return strong.head(top_n).reset_index(drop=True), meta

    remaining = df[~df["PAIR"].isin(set(strong["PAIR"]))].copy()

    need = top_n - len(strong)

    if BALANCED_FALLBACK:
        # En yakın adaylar:
        # LONG adayları: RAW 90'a en yakın (yüksekten düşüğe)
        cand_long = remaining[remaining["RAW"] > STRONG_SHORT_MAX].copy()
        cand_long["dist_long"] = (STRONG_LONG_MIN - cand_long["RAW"]).abs()
        cand_long = cand_long.sort_values(["dist_long", "QV_24H"], ascending=[True, False])

        # SHORT adayları: RAW 10'a en yakın (düşükten yukarı)
        cand_short = remaining[remaining["RAW"] < STRONG_LONG_MIN].copy()
        cand_short["dist_short"] = (cand_short["RAW"] - STRONG_SHORT_MAX).abs()
        cand_short = cand_short.sort_values(["dist_short", "QV_24H"], ascending=[True, False])

        half = top_n // 2  # 10
        # Strong’dan gelenler bir tarafı şişirebilir; o yüzden önce strong ekledik, şimdi "yakın" ile dengeleyeceğiz
        # Ama yine de hedef: tabloda LONG/SHORT karışık dursun.

        take_long = max(0, half - int((strong["YÖN"] == "LONG").sum()))
        take_short = max(0, half - int((strong["YÖN"] == "SHORT").sum()))

        # önce açığı kapat
        pick_long = cand_long.head(take_long)
        pick_short = cand_short.head(take_short)

        picked_pairs = set(strong["PAIR"]).union(set(pick_long["PAIR"])).union(set(pick_short["PAIR"]))
        left_need = top_n - len(picked_pairs)

        # kalan boşluk: "en yakın" mantığıyla doldur (long için 90'a yakın, short için 10'a yakın)
        rest = remaining[~remaining["PAIR"].isin(picked_pairs)].copy()
        rest["near_score"] = np.where(
            rest["RAW"] >= 50,
            (STRONG_LONG_MIN - rest["RAW"]).abs(),   # LONG tarafı: 90'a yakın
            (rest["RAW"] - STRONG_SHORT_MAX).abs(),  # SHORT tarafı: 10'a yakın
        )
        rest = rest.sort_values(["near_score", "QV_24H"], ascending=[True, False]).head(left_need)

        out = pd.concat([strong, pick_long, pick_short, rest], ignore_index=True)
        out = out.drop_duplicates(subset=["PAIR"], keep="first").head(top_n).reset_index(drop=True)
        meta["mode"] = "strong_plus_balanced_near"
        return out, meta

    # fallback: RAW top doldur
    remaining = remaining.sort_values(["RAW", "QV_24H"], ascending=[False, False]).head(need)
    out = pd.concat([strong, remaining], ignore_index=True).head(top_n).reset_index(drop=True)
    meta["mode"] = "strong_plus_top_raw"
    return out, meta


# =============================
# STYLING
# =============================
def style_table(df: pd.DataFrame):
    def dir_style(v):
        v = str(v)
        if v == "LONG":
            return "background-color:#064e3b; color:#ffffff; font-weight:800;"
        if v == "SHORT":
            return "background-color:#7f1d1d; color:#ffffff; font-weight:800;"
        return ""

    def raw_style(v):
        try:
            x = float(v)
        except Exception:
            return ""
        if x >= STRONG_LONG_MIN:
            return "background-color:#006400; color:#ffffff; font-weight:900;"
        if x <= STRONG_SHORT_MAX:
            return "background-color:#8B0000; color:#ffffff; font-weight:900;"
        return "background-color:#0b1220; color:#e6edf3; font-weight:800;"

    fmt = {"FİYAT": "{:.4f}", "SKOR": "{:.0f}", "RAW": "{:.0f}", "QV_24H": "{:,.0f}"}

    return (
        df.style.format(fmt)
        .map(dir_style, subset=["YÖN"])
        .map(raw_style, subset=["SKOR"])
        .map(raw_style, subset=["RAW"])
        .set_properties(**{"border-color": "#1f2a37"})
    )


# =============================
# SCAN
# =============================
def run_scan_all(timeframe: str, limit: int) -> tuple[pd.DataFrame, dict]:
    ex = make_exchange()
    symbols = load_usdt_spot_symbols()

    tickers = safe_fetch_tickers(ex, symbols)

    qv_map = {}
    filtered = []
    for s in symbols:
        qv = quote_volume_usdt(tickers.get(s))
        qv_map[s] = qv
        if qv >= MIN_QV_USDT:
            filtered.append(s)

    scan_symbols = filtered if len(filtered) >= 50 else symbols
    need = max(SMA_PERIOD, BB_PERIOD, RSI_PERIOD, 35) + 5

    rows = []
    errors = 0

    for symbol in scan_symbols:
        try:
            ohlcv = safe_fetch_ohlcv(ex, symbol, timeframe, limit)
            if not ohlcv or len(ohlcv) < need:
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            close = df["close"].astype(float)

            sma20_s = sma(close, SMA_PERIOD)
            _, bb_up_s, bb_low_s = bollinger_bands(close, BB_PERIOD, BB_STD)
            rsi_s = rsi_wilder(close, RSI_PERIOD)
            macd_h_s = macd_hist(close)

            last_close = float(close.iloc[-1])
            last_sma20 = float(sma20_s.iloc[-1])
            last_rsi = float(rsi_s.iloc[-1])
            last_low = float(bb_low_s.iloc[-1])
            last_up = float(bb_up_s.iloc[-1])
            last_macd_h = float(macd_h_s.iloc[-1])

            if any(np.isnan([last_sma20, last_rsi, last_low, last_up, last_macd_h])):
                continue

            raw = score_raw(last_close, last_sma20, last_rsi, last_low, last_up, last_macd_h)
            skor = raw

            rows.append(
                {
                    "PAIR": symbol,
                    "YÖN": direction_from_raw(raw),
                    "COIN": symbol.split("/")[0],
                    "SKOR": int(skor),
                    "FİYAT": last_close,
                    "RAW": int(raw),
                    "QV_24H": float(qv_map.get(symbol, 0.0)),
                }
            )

        except (ccxt.RequestTimeout, ccxt.NetworkError, ccxt.ExchangeError):
            errors += 1
        except Exception:
            errors += 1

        time.sleep(0.03)

    out = pd.DataFrame(rows)
    meta = {"universe_count": len(symbols), "filtered_count": len(scan_symbols), "errors_count": errors}
    if out.empty:
        return out, meta

    out = out.sort_values(["RAW", "QV_24H"], ascending=[False, False]).reset_index(drop=True)
    return out, meta


# =============================
# UI
# =============================
st.set_page_config(page_title="KuCoin PRO Sniper", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.0rem; }
h1, h2, h3, h4, h5, h6, p, span, div { color: #e6edf3; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
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

try_autorefresh(interval_ms=AUTO_REFRESH_SEC * 1000, key="sniper_auto")

now_ist = datetime.now(IST_TZ)

left, right = st.columns([2, 1], vertical_alignment="center")
with left:
    st.title("🎯 KuCoin PRO Sniper — Auto (LONG + SHORT)")
    st.caption(
        f"TF={TF} • STRONG: RAW≥{STRONG_LONG_MIN} LONG / RAW≤{STRONG_SHORT_MAX} SHORT • "
        f"Fallback: {'10 LONG + 10 SHORT (en yakın)' if BALANCED_FALLBACK else 'Top RAW'} • Auto: {AUTO_REFRESH_SEC}s"
    )
with right:
    st.markdown(
        f"""
<div style="text-align:right; padding-top: 8px;">
  <div style="font-size: 12px; opacity: 0.85;">Istanbul Time</div>
  <div style="font-size: 18px; font-weight: 800;">{now_ist.strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# Scan
with st.spinner("⏳ KuCoin USDT spot evreni taranıyor… (auto refresh açık)"):
    df_all, meta_scan = run_scan_all(timeframe=TF, limit=CANDLE_LIMIT)
    if "last_df" not in st.session_state or df_all is not None:
        st.session_state["last_df"] = df_all
        st.session_state["last_meta_scan"] = meta_scan
        st.session_state["last_scan_ist"] = datetime.now(IST_TZ)

df_all = st.session_state.get("last_df")
meta_scan = st.session_state.get("last_meta_scan") or {}
last_scan = st.session_state.get("last_scan_ist")

# Metrics
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    st.metric("Son Tarama (IST)", last_scan.strftime("%H:%M:%S") if last_scan else "—")
with c2:
    st.metric("Evren (USDT spot)", str(meta_scan.get("universe_count", 0)))
with c3:
    st.metric("Taranan (min QV filtresi)", str(meta_scan.get("filtered_count", 0)))
with c4:
    st.metric("Hata/Timeout", str(meta_scan.get("errors_count", 0)))

st.write("")
st.subheader("🎯 SNIPER TABLO")

if not isinstance(df_all, pd.DataFrame) or df_all.empty:
    st.warning("Aday yok (network/KuCoin). Bir sonraki otomatik yenilemede tekrar dener.")
else:
    df_top, meta_table = build_sniper_table(df_all, TOP_ROWS)

    sl = meta_table.get("strong_long", 0)
    ss = meta_table.get("strong_short", 0)
    mode = meta_table.get("mode", "")

    if sl + ss > 0:
        st.success(f"✅ STRONG bulundu — LONG:{sl} • SHORT:{ss} • Mod: {mode}")
    else:
        st.info(f"ℹ️ Şu an STRONG yok. En yakın adaylar (dengeli) gösteriliyor — Mod: {mode}")

    df_show = df_top[["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H"]].copy()
    st.dataframe(style_table(df_show), width="stretch", height=720)
