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
# FIXED CONFIG (no sidebar)
# =============================
IST_TZ = ZoneInfo("Europe/Istanbul")

TIMEFRAME = "15m"
CANDLE_LIMIT = 200

AUTO_REFRESH_SEC = 240

# Universe & filters
MIN_QV_USDT = 200_000      # liquidity filter (quote volume)
MAX_SCAN_UNIVERSE = 450    # scan top by liquidity after filtering
MAX_SPREAD_PCT = 0.35      # spread filter (%). Harder => smaller number (0.20 stricter)

# Indicators
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIOD = 20
ADX_PERIOD = 14
ATR_PERIOD = 14

# Scoring
RAW_STRONG_LONG = 90
RAW_STRONG_SHORT = 10
SCORE_STEP = 5  # score rounds to 0,5,10...

# Output mode
TOP_SNIPER = 3  # "Only best 3" mode (automatic)

# Exclude bases (stable/fiat/pegged-ish list to keep table clean)
EXCLUDED_BASE = {
    "USDT", "USDC", "TUSD", "USDE", "DAI", "FDUSD", "USDP", "BUSD",
    "EUR", "TRY", "GBP", "JPY", "CHF", "AUD", "CAD",
    "PAXG",  # gold-pegged (optional; remove if you want)
}

# =============================
# Exchange
# =============================
def make_exchange() -> ccxt.kucoin:
    return ccxt.kucoin(
        {
            "enableRateLimit": True,
            "timeout": 20000,  # ms
        }
    )

# =============================
# Pure pandas/numpy indicators
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

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_s = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (
        pd.Series(plus_dm, index=high.index)
        .ewm(alpha=1.0 / period, adjust=False, min_periods=period)
        .mean()
        / atr_s.replace(0.0, np.nan)
    )
    minus_di = 100.0 * (
        pd.Series(minus_dm, index=high.index)
        .ewm(alpha=1.0 / period, adjust=False, min_periods=period)
        .mean()
        / atr_s.replace(0.0, np.nan)
    )

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx_v = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx_v.fillna(0.0)

# =============================
# Helpers
# =============================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def round_step(x: float, step: int) -> int:
    return int(np.round(x / step) * step)

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

        base = (m.get("base") or "").upper().strip()
        if base in EXCLUDED_BASE:
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

def get_qv_usdt(t: dict) -> float:
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

def safe_spread_pct(ex: ccxt.Exchange, symbol: str) -> float | None:
    """
    Return spread percent using orderbook best bid/ask.
    None means unable.
    """
    try:
        ob = ex.fetch_order_book(symbol, limit=5)
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return None
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        if mid <= 0:
            return None
        spr = ((best_ask - best_bid) / mid) * 100.0
        return float(spr)
    except Exception:
        return None

# =============================
# Scoring (kademeli + filtreli)
# RAW: 0..100 (long bias)
# SCORE: 0..100 (strength) = max(RAW, 100-RAW)
# =============================
def compute_raw(
    close: float,
    sma20_v: float,
    rsi_v: float,
    bb_low: float,
    bb_up: float,
    adx_v: float,
    atr_pct: float,
) -> float:
    raw = 50.0

    # 1) Trend (±20) scaled by distance vs SMA20 (cap at 2%)
    if sma20_v > 0:
        dist = (close - sma20_v) / sma20_v
        raw += clamp(dist / 0.02, -1.0, 1.0) * 20.0

    # 2) Momentum (±40) RSI around 50; 35 => +40, 65 => -40
    raw += clamp((50.0 - rsi_v) / 15.0, -1.0, 1.0) * 40.0

    # 3) Volatility (±40) how far outside Bollinger, scaled by band width
    width = bb_up - bb_low
    if width > 0:
        if close < bb_low:
            z = (bb_low - close) / width
            raw += clamp(z, 0.0, 1.0) * 40.0
        elif close > bb_up:
            z = (close - bb_up) / width
            raw -= clamp(z, 0.0, 1.0) * 40.0

    # ---- Filters to reduce fake extremes ----
    # ADX: if weak trend, pull toward 50
    if adx_v < 25.0:
        k = clamp(adx_v / 25.0, 0.0, 1.0)
        raw = 50.0 + (raw - 50.0) * k

    # ATR%: if too spiky, pull toward 50
    if atr_pct > 5.0:
        k = 1.0 - clamp((atr_pct - 5.0) / 7.0, 0.0, 1.0) * 0.6
        raw = 50.0 + (raw - 50.0) * k

    return clamp(raw, 0.0, 100.0)

def raw_to_dir_score(raw: float) -> tuple[str, int, int]:
    score = max(raw, 100.0 - raw)
    score_i = round_step(score, SCORE_STEP)
    raw_i = round_step(raw, SCORE_STEP)
    direction = "LONG" if raw >= 50.0 else "SHORT"
    return direction, int(score_i), int(raw_i)

# =============================
# UI Styling
# =============================
def inject_css():
    st.markdown(
        """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
h1, h2, h3, h4, h5, h6, p, span, div { color: #e6edf3; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
</style>
""",
        unsafe_allow_html=True,
    )

def style_table(df: pd.DataFrame):
    def dir_bg(v):
        if v == "LONG":
            return "background-color:#064e3b;color:#ffffff;font-weight:800;"
        if v == "SHORT":
            return "background-color:#7f1d1d;color:#ffffff;font-weight:800;"
        return ""

    def score_bg(v):
        try:
            v = float(v)
        except Exception:
            return ""
        if v >= 90:
            return "background-color:#006400;color:#ffffff;font-weight:900;"
        return "background-color:#0b1220;color:#e6edf3;font-weight:800;"

    fmt = {
        "FİYAT": "{:.4f}",
        "QV_24H": "{:,.0f}",
        "SPREAD%": "{:.2f}",
    }

    return (
        df.style.format(fmt)
        .applymap(dir_bg, subset=["YÖN"])
        .applymap(score_bg, subset=["SKOR"])
        .set_properties(**{"border-color": "#1f2a37"})
    )

def dataframe_render(styler, height=760):
    try:
        st.dataframe(styler, width="stretch", height=height)
    except Exception:
        st.dataframe(styler, use_container_width=True, height=height)

# =============================
# Scan logic
# =============================
def build_universe(ex: ccxt.Exchange, symbols: list[str]) -> tuple[list[str], dict]:
    tickers = safe_fetch_tickers(ex, symbols)
    ranked = []
    for s in symbols:
        qv = get_qv_usdt(tickers.get(s))
        if qv >= MIN_QV_USDT:
            ranked.append((s, qv))
    ranked.sort(key=lambda x: x[1], reverse=True)
    universe = [s for s, _ in ranked[:MAX_SCAN_UNIVERSE]]
    return universe, tickers

def pick_sniper(df_all: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if df_all.empty:
        return df_all, "NO_DATA"

    strong = df_all[(df_all["RAW"] >= RAW_STRONG_LONG) | (df_all["RAW"] <= RAW_STRONG_SHORT)].copy()
    if not strong.empty:
        out = strong.sort_values(["SKOR", "QV_24H"], ascending=[False, False]).head(TOP_SNIPER)
        return out.reset_index(drop=True), "STRONG_TOP3"

    # fallback: closest to extremes (best score)
    out = df_all.sort_values(["SKOR", "QV_24H"], ascending=[False, False]).head(TOP_SNIPER)
    return out.reset_index(drop=True), "FALLBACK_TOP3"

def run_scan() -> tuple[pd.DataFrame, dict]:
    ex = make_exchange()
    symbols = load_usdt_spot_symbols()
    if not symbols:
        return pd.DataFrame(), {"universe": 0, "scored": 0, "filtered_spread": 0}

    universe, tickers = build_universe(ex, symbols)
    if not universe:
        return pd.DataFrame(), {"universe": 0, "scored": 0, "filtered_spread": 0}

    rows = []
    scored = 0
    filtered_spread = 0

    progress = st.progress(0, text="Piyasalar taranıyor… (KuCoin / USDT Spot)")
    status = st.empty()

    total = len(universe)
    for i, sym in enumerate(universe, start=1):
        if i == 1 or i % 10 == 0 or i == total:
            progress.progress(int((i / total) * 100), text=f"Taranıyor: {sym} ({i}/{total})")

        try:
            # Spread filter (orderbook)
            spr = safe_spread_pct(ex, sym)
            if spr is None:
                # if no orderbook, skip (sniper mode)
                filtered_spread += 1
                continue
            if spr > MAX_SPREAD_PCT:
                filtered_spread += 1
                continue

            ohlcv = safe_fetch_ohlcv(ex, sym, TIMEFRAME, CANDLE_LIMIT)
            if not ohlcv or len(ohlcv) < max(SMA_PERIOD, BB_PERIOD, RSI_PERIOD, ADX_PERIOD, ATR_PERIOD) + 5:
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)

            sma20_s = sma(close, SMA_PERIOD)
            _, bb_up_s, bb_low_s = bollinger(close, BB_PERIOD, BB_STD)
            rsi_s = rsi_wilder(close, RSI_PERIOD)
            adx_s = adx(high, low, close, ADX_PERIOD)
            atr_s = atr(high, low, close, ATR_PERIOD)

            last_close = float(close.iloc[-1])
            last_sma20 = float(sma20_s.iloc[-1])
            last_rsi = float(rsi_s.iloc[-1])
            last_bb_low = float(bb_low_s.iloc[-1])
            last_bb_up = float(bb_up_s.iloc[-1])
            last_adx = float(adx_s.iloc[-1])
            last_atr = float(atr_s.iloc[-1])

            if any(np.isnan([last_sma20, last_rsi, last_bb_low, last_bb_up, last_adx, last_atr])) or last_close <= 0:
                continue

            atr_pct = (last_atr / last_close) * 100.0 if last_close else 0.0

            raw = compute_raw(
                close=last_close,
                sma20_v=last_sma20,
                rsi_v=last_rsi,
                bb_low=last_bb_low,
                bb_up=last_bb_up,
                adx_v=last_adx,
                atr_pct=atr_pct,
            )

            direction, score_i, raw_i = raw_to_dir_score(raw)

            t = tickers.get(sym) or {}
            qv = get_qv_usdt(t)

            rows.append(
                {
                    "YÖN": direction,
                    "COIN": sym.replace("/USDT", ""),
                    "SKOR": score_i,
                    "FİYAT": float(last_close),
                    "RAW": raw_i,
                    "SPREAD%": float(spr),
                    "QV_24H": float(qv),
                }
            )
            scored += 1

        except (ccxt.RequestTimeout, ccxt.NetworkError):
            status.warning(f"Network/Timeout: {sym}")
        except ccxt.ExchangeError:
            pass
        except Exception:
            pass

        time.sleep(0.03)

    progress.empty()
    status.empty()

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"universe": len(universe), "scored": 0, "filtered_spread": filtered_spread}

    out = out.sort_values(["SKOR", "QV_24H"], ascending=[False, False]).reset_index(drop=True)
    return out, {"universe": len(universe), "scored": scored, "filtered_spread": filtered_spread}

# =============================
# Streamlit App
# =============================
st.set_page_config(page_title="KuCoin PRO Sniper (Spot)", layout="wide")
inject_css()

# Auto refresh
try:
    st.autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
except Exception:
    try:
        st.experimental_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
    except Exception:
        pass

now_ist = datetime.now(IST_TZ)

st.markdown(
    f"""
<div style="display:flex; justify-content:space-between; align-items:flex-end; gap:12px;">
  <div>
    <div style="font-size:42px; font-weight:900; line-height:1.05;">KuCoin PRO Sniper</div>
    <div style="opacity:.85; margin-top:6px;">
      TF={TIMEFRAME} · SNIPER=Top {TOP_SNIPER} · STRONG: RAW≥{RAW_STRONG_LONG} / RAW≤{RAW_STRONG_SHORT} ·
      QV_24H≥{MIN_QV_USDT:,} · Spread≤{MAX_SPREAD_PCT:.2f}% · Auto refresh: {AUTO_REFRESH_SEC}s
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:12px; opacity:.85;">Istanbul Time</div>
    <div style="font-size:18px; font-weight:800;">{now_ist.strftime('%Y-%m-%d %H:%M:%S')}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

with st.spinner("🔄 Tarama başlıyor… (evren hazırlanıyor, spread kontrol ediliyor, mumlar çekiliyor)"):
    df_all, meta = run_scan()

# Status bar
if meta.get("universe", 0) == 0:
    st.warning("KuCoin USDT spot evreni alınamadı (network/limit). Biraz sonra otomatik yenilenecek.")
elif meta.get("scored", 0) == 0:
    st.info(
        f"Evren (filtreli): {meta.get('universe',0)} · Skorlanan: 0 · Spread filtrelenen: {meta.get('filtered_spread',0)} "
        "→ Network/KuCoin limit olabilir. Yenilenmeyi bekle."
    )
else:
    st.success(
        f"✅ Tarama bitti · Evren (filtreli): {meta.get('universe',0)} · Skorlanan: {meta.get('scored',0)} · "
        f"Spread filtrelenen: {meta.get('filtered_spread',0)}"
    )

st.write("")

# Final sniper table
if df_all.empty:
    st.info("Tablo için veri yok. Bir sonraki auto refresh ile dolacaktır.")
else:
    df_sniper, mode = pick_sniper(df_all)

    if mode == "STRONG_TOP3":
        st.markdown("### 🎯 SNIPER TABLO (STRONG — Top 3)")
    else:
        st.markdown("### 🎯 SNIPER TABLO (STRONG yok — En iyi Top 3 aday)")

    dataframe_render(style_table(df_sniper), height=420)

    st.caption(
        "SKOR = sinyal gücü (LONG/SHORT fark etmez), RAW = yön eğilimi (0→SHORT, 100→LONG). "
        "Bu sürüm Spread + Likidite + ADX/ATR ile ‘fake 100’leri ciddi azaltır."
    )
