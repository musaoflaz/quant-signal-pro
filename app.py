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
# FIXED SETTINGS (no sidebar)
# =============================
IST_TZ = ZoneInfo("Europe/Istanbul")

BASE_TF = "15m"
BASE_LIMIT = 220

# Scan all USDT spot pairs
HARD_MAX_SYMBOLS = 99999

# Score engine
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIOD = 20

# PRO regime filters
EMA_FAST = 50
EMA_SLOW = 200
ADX_PERIOD = 14

# STRONG gates (RAW 0–100)
STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10

# Fallback balance
FALLBACK_LONG = 10
FALLBACK_SHORT = 10

# Auto refresh
AUTO_REFRESH_SEC = 240  # 4 dk

# HARD MODE
PRO_MODE = True

# Spread filter (finalists only)
REQUIRE_SPREAD = True
MAX_SPREAD_PCT = 0.55

# Asset trend strength on 1h/4h
REQUIRE_REGIME_1H = True
REQUIRE_REGIME_4H = True

REQUIRE_ADX_1H = True
REQUIRE_ADX_4H = True
ADX1_MIN = 22.0
ADX4_MIN = 20.0

# 2-candle confirmation
REQUIRE_2CANDLE_CONFIRM = True

# BTC/ETH regime bias (never empties table)
USE_BTC_ETH_REGIME_GATE = True
REGIME_TF_1 = "1h"
REGIME_TF_2 = "4h"
REGIME_ADX_MIN_1H = 18.0
REGIME_ADX_MIN_4H = 16.0


# =============================
# Streamlit helpers (compat)
# =============================
def safe_autorefresh(interval_ms: int, key: str):
    try:
        return st.autorefresh(interval=interval_ms, key=key)
    except Exception:
        try:
            return st.experimental_autorefresh(interval=interval_ms, key=key)
        except Exception:
            return None


def safe_dataframe(data, height: int = 600):
    # New Streamlit versions prefer width="stretch"
    try:
        return st.dataframe(data, width="stretch", height=height)
    except TypeError:
        return st.dataframe(data, use_container_width=True, height=height)


# =============================
# Indicator math (pure pandas/numpy)
# =============================
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


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


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_dm_s = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_s = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (plus_dm_s / atr.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm_s / atr.replace(0.0, np.nan))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)


# =============================
# Score engine (RAW 0–100)
# =============================
def raw_sniper_score(close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float) -> int:
    score = 50

    # Trend (20)
    score += 20 if close > sma20_v else -20

    # Momentum (40)
    if rsi_v < 35:
        score += 40
    elif rsi_v > 65:
        score -= 40

    # Volatility (40)
    if close <= bb_low:
        score += 40
    elif close >= bb_up:
        score -= 40

    return int(max(0, min(100, score)))


def direction_from_raw(raw_score: int) -> str:
    return "LONG" if raw_score >= 50 else "SHORT"


def strength_from_raw(raw_score: int) -> int:
    # display score always “high = strong”
    return int(raw_score) if raw_score >= 50 else int(100 - raw_score)


def is_strong(raw_score: int) -> bool:
    return (raw_score >= STRONG_LONG_MIN) or (raw_score <= STRONG_SHORT_MAX)


# =============================
# KuCoin / CCXT helpers
# =============================
def make_exchange() -> ccxt.kucoin:
    return ccxt.kucoin({"enableRateLimit": True, "timeout": 20000})


@st.cache_data(show_spinner=False, ttl=900)
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
        syms.append(sym)
    return sorted(set(syms))


def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def safe_fetch_tickers_all(ex: ccxt.Exchange) -> dict:
    try:
        return ex.fetch_tickers()
    except Exception:
        return {}


def quote_volume_usdt(ticker: dict) -> float:
    if not ticker or not isinstance(ticker, dict):
        return 0.0
    try:
        qv = ticker.get("quoteVolume", None)
        if qv is not None:
            return float(qv)
    except Exception:
        pass
    try:
        bv = ticker.get("baseVolume", None)
        last = ticker.get("last", None)
        if bv is not None and last is not None:
            return float(bv) * float(last)
    except Exception:
        pass
    return 0.0


def safe_fetch_orderbook_spread_pct(ex: ccxt.Exchange, symbol: str) -> float | None:
    try:
        ob = ex.fetch_order_book(symbol, limit=5)
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return None
        bid = float(bids[0][0])
        ask = float(asks[0][0])
        if bid <= 0 or ask <= 0 or ask <= bid:
            return None
        mid = (ask + bid) / 2.0
        return ((ask - bid) / mid) * 100.0
    except Exception:
        return None


# =============================
# Base row builder (15m)
# =============================
def build_base_row(symbol: str, ohlcv: list) -> dict | None:
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    need = max(SMA_PERIOD, BB_PERIOD, RSI_PERIOD) + 10
    if len(df) < need:
        return None

    close = df["close"].astype(float)

    sma20_s = sma(close, SMA_PERIOD)
    _, bb_up_s, bb_low_s = bollinger_bands(close, BB_PERIOD, BB_STD)
    rsi_s = rsi_wilder(close, RSI_PERIOD)

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])

    last_sma20 = float(sma20_s.iloc[-1])
    last_rsi = float(rsi_s.iloc[-1])
    last_low = float(bb_low_s.iloc[-1])
    last_up = float(bb_up_s.iloc[-1])

    prev_low = float(bb_low_s.iloc[-2])
    prev_up = float(bb_up_s.iloc[-2])
    prev_rsi = float(rsi_s.iloc[-2])

    if any(np.isnan([last_sma20, last_rsi, last_low, last_up, prev_low, prev_up, prev_rsi])):
        return None

    raw = raw_sniper_score(last_close, last_sma20, last_rsi, last_low, last_up)
    yon = direction_from_raw(raw)
    skor = strength_from_raw(raw)

    confirm_ok = True
    if PRO_MODE and REQUIRE_2CANDLE_CONFIRM:
        if yon == "LONG":
            confirm_ok = (prev_close <= prev_low) and (last_close > last_low) and (last_rsi > prev_rsi)
        else:
            confirm_ok = (prev_close >= prev_up) and (last_close < last_up) and (last_rsi < prev_rsi)

    coin = symbol.replace("/USDT", "")
    return {
        "SYMBOL": symbol,
        "COIN": coin,
        "YÖN": yon,
        "RAW": int(raw),
        "SKOR": int(skor),
        "FİYAT": float(last_close),
        "RSI14": float(last_rsi),
        "SMA20": float(last_sma20),
        "BB_LOWER": float(last_low),
        "BB_UPPER": float(last_up),
        "CONFIRM_OK": bool(confirm_ok),
    }


# =============================
# BTC/ETH regime (bias only)
# =============================
def regime_for_symbol(ex: ccxt.Exchange, symbol: str) -> str:
    try:
        o1 = safe_fetch_ohlcv(ex, symbol, REGIME_TF_1, 320)
        o4 = safe_fetch_ohlcv(ex, symbol, REGIME_TF_2, 320)
        if not o1 or not o4:
            return "NA"

        def calc_reg(ohlcv: list, adx_min: float) -> str:
            d = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            h = d["high"].astype(float)
            l = d["low"].astype(float)
            c = d["close"].astype(float)

            e50 = ema(c, EMA_FAST)
            e200 = ema(c, EMA_SLOW)
            ax = adx_wilder(h, l, c, ADX_PERIOD)

            last_close = float(c.iloc[-1])
            last_e50 = float(e50.iloc[-1]) if not np.isnan(e50.iloc[-1]) else np.nan
            last_e200 = float(e200.iloc[-1]) if not np.isnan(e200.iloc[-1]) else np.nan
            last_adx = float(ax.iloc[-1]) if not np.isnan(ax.iloc[-1]) else 0.0

            if np.isnan(last_e50) or np.isnan(last_e200):
                return "NEUTRAL"

            bull = (last_close > last_e200) and (last_e50 > last_e200) and (last_adx >= adx_min)
            bear = (last_close < last_e200) and (last_e50 < last_e200) and (last_adx >= adx_min)

            if bull:
                return "BULL"
            if bear:
                return "BEAR"
            return "NEUTRAL"

        r1 = calc_reg(o1, REGIME_ADX_MIN_1H)
        r4 = calc_reg(o4, REGIME_ADX_MIN_4H)

        if r1 == "BULL" and r4 == "BULL":
            return "BULL"
        if r1 == "BEAR" and r4 == "BEAR":
            return "BEAR"
        return "NEUTRAL"
    except Exception:
        return "NA"


def btc_eth_bias(btc_reg: str, eth_reg: str) -> tuple[str, str]:
    if btc_reg == "BULL" and eth_reg == "BULL":
        return "LONG", "BTC & ETH BULL"
    if btc_reg == "BEAR" and eth_reg == "BEAR":
        return "SHORT", "BTC & ETH BEAR"
    if btc_reg == "BULL" or eth_reg == "BULL":
        return "LONG", "Regime: BULL-ish (one strong)"
    if btc_reg == "BEAR" or eth_reg == "BEAR":
        return "SHORT", "Regime: BEAR-ish (one strong)"
    return "BOTH", "BTC/ETH neutral (bias yok)"


# =============================
# PRO gates on finalists only
# =============================
def pro_pass(ex: ccxt.Exchange, symbol: str, want_dir: str) -> tuple[bool, dict]:
    info = {"SPREAD%": np.nan, "ADX_1H": np.nan, "ADX_4H": np.nan}

    if REQUIRE_SPREAD:
        sp = safe_fetch_orderbook_spread_pct(ex, symbol)
        info["SPREAD%"] = sp if sp is not None else np.nan
        if sp is None or sp > MAX_SPREAD_PCT:
            return False, info

    def tf_gate(tf: str, require_regime: bool, require_adx: bool, adx_min: float) -> bool:
        o = safe_fetch_ohlcv(ex, symbol, tf, 320)
        if not o or len(o) < max(EMA_SLOW, ADX_PERIOD) + 10:
            return False

        d = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
        h = d["high"].astype(float)
        l = d["low"].astype(float)
        c = d["close"].astype(float)

        e50 = ema(c, EMA_FAST)
        e200 = ema(c, EMA_SLOW)
        last_close = float(c.iloc[-1])
        last_e50 = float(e50.iloc[-1]) if not np.isnan(e50.iloc[-1]) else np.nan
        last_e200 = float(e200.iloc[-1]) if not np.isnan(e200.iloc[-1]) else np.nan

        if require_regime:
            if np.isnan(last_e50) or np.isnan(last_e200):
                return False
            if want_dir == "LONG":
                if not (last_close > last_e200 and last_e50 > last_e200):
                    return False
            else:
                if not (last_close < last_e200 and last_e50 < last_e200):
                    return False

        if require_adx:
            ax = adx_wilder(h, l, c, ADX_PERIOD)
            last_adx = float(ax.iloc[-1]) if not np.isnan(ax.iloc[-1]) else 0.0
            if tf == "1h":
                info["ADX_1H"] = last_adx
            else:
                info["ADX_4H"] = last_adx
            if last_adx < adx_min:
                return False

        return True

    if REQUIRE_REGIME_1H or REQUIRE_ADX_1H:
        if not tf_gate("1h", REQUIRE_REGIME_1H, REQUIRE_ADX_1H, ADX1_MIN):
            return False, info

    if REQUIRE_REGIME_4H or REQUIRE_ADX_4H:
        if not tf_gate("4h", REQUIRE_REGIME_4H, REQUIRE_ADX_4H, ADX4_MIN):
            return False, info

    return True, info


# =============================
# Styling
# =============================
def style_table(df: pd.DataFrame):
    def dir_style(v):
        s = str(v)
        if s == "LONG":
            return "background-color: #064e3b; color: #ffffff; font-weight: 800;"
        if s == "SHORT":
            return "background-color: #7f1d1d; color: #ffffff; font-weight: 800;"
        return ""

    def score_style(v):
        try:
            x = float(v)
        except Exception:
            return ""
        if x >= 95:
            return "background-color: #006400; color: #ffffff; font-weight: 900;"
        if x >= 90:
            return "background-color: #0f3d0f; color: #ffffff; font-weight: 800;"
        return "background-color: #111827; color: #e5e7eb;"

    fmt = {"FİYAT": "{:.4f}", "SKOR": "{:.0f}", "RAW": "{:.0f}", "QV_24H": "{:,.0f}"}
    return (
        df.style.format(fmt)
        .applymap(dir_style, subset=["YÖN"])
        .applymap(score_style, subset=["SKOR"])
        .set_properties(**{"border-color": "#1f2a37"})
    )


# =============================
# UI (no sidebar) + BIG LOADING PANEL
# =============================
st.set_page_config(page_title="KuCoin PRO Sniper — Simple", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.0rem; }
h1, h2, h3, h4, h5, h6, p, span, div { color: #e6edf3; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }

.loading-card{
  border: 1px solid #1f2a37;
  background: #0f172a;
  padding: 14px 14px;
  border-radius: 14px;
  display:flex;
  align-items:center;
  gap: 12px;
}
.spinner{
  width: 22px; height:22px;
  border: 3px solid rgba(255,255,255,0.18);
  border-top: 3px solid rgba(255,255,255,0.85);
  border-radius: 50%;
  animation: spin 0.9s linear infinite;
}
@keyframes spin { 0%{transform:rotate(0deg)} 100%{transform:rotate(360deg)} }
.small-muted{ opacity:0.8; font-size: 12px; }
.big{ font-size: 16px; font-weight: 800; }
</style>
""",
    unsafe_allow_html=True,
)

safe_autorefresh(interval_ms=AUTO_REFRESH_SEC * 1000, key="auto_refresh")

left, right = st.columns([2, 1], vertical_alignment="center")
with left:
    st.title("KuCoin PRO Sniper — BTC/ETH Rejim Filtreli (Sade)")
    st.caption(
        f"TF={BASE_TF} • STRONG: RAW>=90 LONG / RAW<=10 SHORT • "
        f"STRONG yoksa: {FALLBACK_LONG} LONG + {FALLBACK_SHORT} SHORT en yakın aday • "
        f"Auto refresh: {AUTO_REFRESH_SEC}s"
    )
with right:
    now_ist = datetime.now(IST_TZ)
    st.markdown(
        f"""
<div style="text-align:right; padding-top: 6px;">
  <div style="font-size: 12px; opacity: 0.85;">Istanbul Time</div>
  <div style="font-size: 18px; font-weight: 700;">{now_ist.strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
""",
        unsafe_allow_html=True,
    )

loading_box = st.empty()
progress = st.progress(0, text="Hazırlanıyor…")
status = st.empty()


def show_loading(title: str, subtitle: str):
    loading_box.markdown(
        f"""
<div class="loading-card">
  <div class="spinner"></div>
  <div>
    <div class="big">{title}</div>
    <div class="small-muted">{subtitle}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def hide_loading():
    loading_box.empty()
    status.empty()
    progress.empty()


def run_scan_with_progress() -> dict:
    ex = make_exchange()

    show_loading("Taranıyor…", "KuCoin USDT Spot evreni hazırlanıyor (tickers + OHLCV).")
    progress.progress(1, text="Piyasalar yükleniyor…")

    # Regime
    if USE_BTC_ETH_REGIME_GATE:
        progress.progress(3, text="BTC/ETH rejimi hesaplanıyor…")
        btc_reg = regime_for_symbol(ex, "BTC/USDT")
        eth_reg = regime_for_symbol(ex, "ETH/USDT")
        bias, bias_msg = btc_eth_bias(btc_reg, eth_reg)
    else:
        btc_reg, eth_reg, bias, bias_msg = "NA", "NA", "BOTH", "Regime gate disabled"

    syms = load_usdt_spot_symbols()
    universe = syms[: min(len(syms), HARD_MAX_SYMBOLS)]

    progress.progress(6, text="Ticker verileri çekiliyor…")
    tickers = safe_fetch_tickers_all(ex)
    qv_map = {s: quote_volume_usdt(tickers.get(s)) for s in universe} if isinstance(tickers, dict) else {s: 0.0 for s in universe}

    # Stage 1: base scan
    rows = []
    total = max(1, len(universe))
    show_loading("Taranıyor…", f"OHLCV çekiliyor • Evren: {len(universe)} coin • TF: {BASE_TF}")
    for i, symbol in enumerate(universe, start=1):
        try:
            o = safe_fetch_ohlcv(ex, symbol, BASE_TF, BASE_LIMIT)
            if not o:
                continue
            r = build_base_row(symbol, o)
            if r is None:
                continue
            r["QV_24H"] = float(qv_map.get(symbol, 0.0))
            rows.append(r)
        except Exception:
            pass

        # progress update (smooth)
        if i % 10 == 0 or i == total:
            pct = 6 + int((i / total) * 62)  # 6..68
            progress.progress(min(68, pct), text=f"OHLCV taranıyor… {i}/{total}")

        time.sleep(0.006)

    df_all = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Bias only
    if not df_all.empty and bias in ("LONG", "SHORT"):
        df_all = df_all[df_all["YÖN"] == bias].copy()

    # Stage 2: strong + pro pass (only candidates)
    progress.progress(72, text="STRONG adayları ayıklanıyor…")
    show_loading("Analiz…", "STRONG adaylar için PRO filtreler (spread + 1h/4h trend/ADX) kontrol ediliyor.")

    strong = pd.DataFrame()
    if not df_all.empty:
        cand = df_all[df_all["RAW"].apply(is_strong)].copy()

        if PRO_MODE and REQUIRE_2CANDLE_CONFIRM and not cand.empty:
            cand = cand[cand["CONFIRM_OK"] == True].copy()

        if PRO_MODE and not cand.empty:
            kept = []
            info_map = {}
            ctotal = len(cand)
            for j, row in enumerate(cand.itertuples(index=False), start=1):
                symbol = getattr(row, "SYMBOL")
                raw = int(getattr(row, "RAW"))
                want_dir = "LONG" if raw >= 50 else "SHORT"
                ok, info = pro_pass(ex, symbol, want_dir)
                if ok:
                    kept.append(symbol)
                    info_map[symbol] = info

                if j % 5 == 0 or j == ctotal:
                    pct = 72 + int((j / max(1, ctotal)) * 23)  # 72..95
                    progress.progress(min(95, pct), text=f"PRO filtre… {j}/{ctotal}")

                time.sleep(0.006)

            if kept:
                strong = cand[cand["SYMBOL"].isin(set(kept))].copy()
                strong["SPREAD%"] = strong["SYMBOL"].map(lambda s: info_map.get(s, {}).get("SPREAD%", np.nan))
                strong["ADX_1H"] = strong["SYMBOL"].map(lambda s: info_map.get(s, {}).get("ADX_1H", np.nan))
                strong["ADX_4H"] = strong["SYMBOL"].map(lambda s: info_map.get(s, {}).get("ADX_4H", np.nan))
        else:
            strong = cand.copy()

    progress.progress(100, text="Tamamlandı ✅")
    time.sleep(0.2)

    return {
        "all": df_all,
        "strong": strong,
        "regime": {"btc": btc_reg, "eth": eth_reg, "bias": bias, "msg": bias_msg},
        "universe_count": int(len(universe)),
        "all_count": int(len(df_all)) if isinstance(df_all, pd.DataFrame) else 0,
        "ts": datetime.now(IST_TZ).strftime("%Y-%m-%d %H:%M:%S"),
    }


try:
    res = run_scan_with_progress()
finally:
    hide_loading()

reg = res.get("regime", {})
bias = reg.get("bias", "BOTH")
msg = reg.get("msg", "")

if USE_BTC_ETH_REGIME_GATE:
    if bias == "LONG":
        st.success(f"✅ REGIME BIAS: LONG • {msg}")
    elif bias == "SHORT":
        st.error(f"✅ REGIME BIAS: SHORT • {msg}")
    else:
        st.warning(f"⚠️ REGIME: NEUTRAL • {msg} (tablo yine dolu gelir)")

st.caption(f"Son tarama: {res.get('ts','—')} • Evren (USDT spot): {res.get('universe_count', 0)} • Aday (bias sonrası): {res.get('all_count', 0)}")

st.subheader("🎯 SNIPER TABLO")

df_strong = res.get("strong")
df_all = res.get("all")

if isinstance(df_strong, pd.DataFrame) and not df_strong.empty:
    show = df_strong.copy()
    show = show.sort_values(["SKOR", "RAW", "QV_24H"], ascending=[False, False, False]).head(20).reset_index(drop=True)

    cols = ["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H"]
    for extra in ["SPREAD%", "ADX_1H", "ADX_4H"]:
        if extra in show.columns:
            cols.append(extra)

    show = show.loc[:, [c for c in cols if c in show.columns]]
    safe_dataframe(style_table(show), height=620)

else:
    st.info("Şu an STRONG yok. En yakın adayları (10 LONG + 10 SHORT) gösteriyorum.")

    if not isinstance(df_all, pd.DataFrame) or df_all.empty:
        st.warning("Aday yok (KuCoin/network). Bir sonraki yenilemede tekrar dener.")
    else:
        base = df_all.copy()

        base["DIST_STRONG"] = np.where(
            base["RAW"] >= 50,
            (STRONG_LONG_MIN - base["RAW"]).clip(lower=0),
            (base["RAW"] - STRONG_SHORT_MAX).clip(lower=0),
        )

        long_edge = (35.0 - base["RSI14"]).clip(lower=0) + (
            ((base["BB_LOWER"] - base["FİYAT"]) / base["FİYAT"]) * 100.0 * 5.0
        ).clip(lower=0)
        short_edge = (base["RSI14"] - 65.0).clip(lower=0) + (
            ((base["FİYAT"] - base["BB_UPPER"]) / base["FİYAT"]) * 100.0 * 5.0
        ).clip(lower=0)
        base["EDGE"] = np.where(base["RAW"] >= 50, long_edge, short_edge)

        longs = (
            base[base["YÖN"] == "LONG"]
            .copy()
            .sort_values(["DIST_STRONG", "EDGE", "QV_24H"], ascending=[True, False, False])
            .head(FALLBACK_LONG)
        )
        shorts = (
            base[base["YÖN"] == "SHORT"]
            .copy()
            .sort_values(["DIST_STRONG", "EDGE", "QV_24H"], ascending=[True, False, False])
            .head(FALLBACK_SHORT)
        )

        near = pd.concat([longs, shorts], ignore_index=True)
        near = near.sort_values(["DIST_STRONG", "EDGE", "QV_24H"], ascending=[True, False, False]).reset_index(drop=True)

        cols = ["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H"]
        near = near.loc[:, [c for c in cols if c in near.columns]]

        safe_dataframe(style_table(near), height=620)
