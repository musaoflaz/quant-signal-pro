# requirements.txt (install these):
# streamlit
# pandas
# numpy
# ccxt

from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import ccxt

# =============================
# CONFIG
# =============================
IST_TZ = ZoneInfo("Europe/Istanbul")

# Base indicators (score engine)
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIOD = 20

# PRO/Strict filters (quality gates)
EMA_FAST = 50
EMA_SLOW = 200
ADX_PERIOD = 14
VOL_SMA_PERIOD = 20

# Strong signal gates (RAW score)
STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10

# Universe stability
DEFAULT_HARD_MAX_SYMBOLS = 700
DEFAULT_BASE_LIMIT = 220

# Defaults: HARD mode
DEFAULT_PRO_MODE = True

DEFAULT_REQUIRE_REGIME_1H = True
DEFAULT_REQUIRE_REGIME_4H = True
DEFAULT_REQUIRE_ADX_1H = True
DEFAULT_REQUIRE_ADX_4H = True

DEFAULT_REQUIRE_VOL_CONFIRM = True
DEFAULT_VOL_MULT = 1.30

DEFAULT_REQUIRE_SPREAD = True
DEFAULT_MAX_SPREAD_PCT = 0.40

DEFAULT_REQUIRE_24H_QUOTEVOL = True
DEFAULT_MIN_QUOTEVOL_USDT = 2_000_000

DEFAULT_REQUIRE_2CANDLE_CONFIRM = True

# Auto-scan
AUTO_SCAN_ON_LOAD = True
DEFAULT_AUTO_SCAN = True
DEFAULT_REFRESH_SEC = 240


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
# Score engine (RAW)
# =============================
def raw_sniper_score(close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float) -> int:
    score = 50
    score += 20 if close > sma20_v else -20

    if rsi_v < 35:
        score += 40
    elif rsi_v > 65:
        score -= 40

    if close <= bb_low:
        score += 40
    elif close >= bb_up:
        score -= 40

    return int(max(0, min(100, score)))


def direction_from_raw(raw_score: int) -> str:
    return "LONG" if raw_score >= 50 else "SHORT"


def strength_from_raw(raw_score: int) -> int:
    return int(raw_score) if raw_score >= 50 else int(100 - raw_score)


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
# Base row build (2-candle confirm)
# =============================
def build_base_row(symbol: str, ohlcv: list, base_tf: str, require_2candle_confirm: bool) -> dict | None:
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    need = max(SMA_PERIOD, BB_PERIOD, RSI_PERIOD, VOL_SMA_PERIOD) + 10
    if len(df) < need:
        return None

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    sma20_s = sma(close, SMA_PERIOD)
    _, bb_up_s, bb_low_s = bollinger_bands(close, BB_PERIOD, BB_STD)
    rsi_s = rsi_wilder(close, RSI_PERIOD)
    vol_sma20 = sma(volume, VOL_SMA_PERIOD)

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])

    last_sma20 = float(sma20_s.iloc[-1])
    last_rsi = float(rsi_s.iloc[-1])
    last_low = float(bb_low_s.iloc[-1])
    last_up = float(bb_up_s.iloc[-1])

    prev_low = float(bb_low_s.iloc[-2])
    prev_up = float(bb_up_s.iloc[-2])
    prev_rsi = float(rsi_s.iloc[-2])

    last_vol = float(volume.iloc[-1])
    last_vol_sma = float(vol_sma20.iloc[-1]) if not np.isnan(vol_sma20.iloc[-1]) else 0.0

    if any(np.isnan([last_sma20, last_rsi, last_low, last_up, prev_low, prev_up, prev_rsi])):
        return None

    raw = raw_sniper_score(last_close, last_sma20, last_rsi, last_low, last_up)
    yon = direction_from_raw(raw)
    skor = strength_from_raw(raw)

    confirm_ok = True
    if require_2candle_confirm:
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
        "VOL_LAST": float(last_vol),
        "VOL_SMA20": float(last_vol_sma),
        "CONFIRM_OK": bool(confirm_ok),
        "BASE_TF": base_tf,
    }


# =============================
# PRO gates (EMA50/EMA200 alignment + ADX + Spread)
# =============================
def pro_pass(
    ex: ccxt.Exchange,
    symbol: str,
    want_dir: str,
    require_regime_1h: bool,
    require_regime_4h: bool,
    require_adx_1h: bool,
    require_adx_4h: bool,
    adx1_min: float,
    adx4_min: float,
    require_spread: bool,
    max_spread_pct: float,
) -> tuple[bool, dict]:
    info = {
        "SPREAD%": np.nan,
        "ADX_1H": np.nan,
        "ADX_4H": np.nan,
    }

    if require_spread:
        sp = safe_fetch_orderbook_spread_pct(ex, symbol)
        info["SPREAD%"] = sp if sp is not None else np.nan
        if sp is None or sp > max_spread_pct:
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

    if require_regime_1h or require_adx_1h:
        if not tf_gate("1h", require_regime_1h, require_adx_1h, adx1_min):
            return False, info

    if require_regime_4h or require_adx_4h:
        if not tf_gate("4h", require_regime_4h, require_adx_4h, adx4_min):
            return False, info

    return True, info


# =============================
# Cooldown (same symbol + direction)
# =============================
def cooldown_allow(symbol: str, direction: str, cooldown_min: int) -> bool:
    if cooldown_min <= 0:
        return True
    key = f"{symbol}|{direction}"
    m = st.session_state.get("shown_cooldown", {})
    last_iso = m.get(key)
    if not last_iso:
        return True
    try:
        last_dt = datetime.fromisoformat(last_iso)
    except Exception:
        return True
    return datetime.now(IST_TZ) - last_dt >= timedelta(minutes=cooldown_min)


def cooldown_mark(symbol: str, direction: str):
    key = f"{symbol}|{direction}"
    m = st.session_state.get("shown_cooldown", {})
    m[key] = datetime.now(IST_TZ).isoformat()
    st.session_state["shown_cooldown"] = m


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

    fmt = {
        "FİYAT": "{:.4f}",
        "SKOR": "{:.0f}",
        "RAW": "{:.0f}",
        "SPREAD%": "{:.2f}",
        "QV_24H": "{:,.0f}",
        "ADX_1H": "{:.1f}",
        "ADX_4H": "{:.1f}",
    }

    return (
        df.style.format(fmt)
        .applymap(dir_style, subset=["YÖN"])
        .applymap(score_style, subset=["SKOR"])
        .set_properties(**{"border-color": "#1f2a37"})
    )


# =============================
# UI
# =============================
st.set_page_config(page_title="KuCoin PRO Sniper (Hard Mode)", layout="wide")
st.markdown(
    """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.0rem; }
h1, h2, h3, h4, h5, h6, p, span, div { color: #e6edf3; }
[data-testid="stSidebar"] { background-color: #0b0f14; border-right: 1px solid #1f2a37; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
</style>
""",
    unsafe_allow_html=True,
)

if "results" not in st.session_state:
    st.session_state["results"] = None
if "last_scan_time" not in st.session_state:
    st.session_state["last_scan_time"] = None
if "boot_scanned" not in st.session_state:
    st.session_state["boot_scanned"] = False
if "shown_cooldown" not in st.session_state:
    st.session_state["shown_cooldown"] = {}

# Header
c1, c2 = st.columns([2, 1], vertical_alignment="center")
with c1:
    st.title("KuCoin PRO Sniper — HARD MODE")
    st.caption("Sniper Mode + LONG/SHORT filtresi + Cooldown. STRONG için RAW>=90 (LONG) / RAW<=10 (SHORT).")
with c2:
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

# Sidebar
with st.sidebar:
    st.subheader("Tarama")

    base_tf = st.selectbox("TF", ["5m", "15m", "30m", "1h"], index=1)
    base_limit = st.slider("Mum sayısı", 160, 500, DEFAULT_BASE_LIMIT, step=20)
    hard_max = st.slider("USDT spot evren (adet)", 150, 1400, DEFAULT_HARD_MAX_SYMBOLS, step=50)

    st.write("---")
    st.subheader("Sniper Ekranı")
    sniper_top = st.slider("Kaç sinyal gösterilsin?", 1, 50, 3, step=1)
    dir_filter = st.selectbox("Yön filtresi", ["HEPSİ", "LONG", "SHORT"], index=0)
    show_near_if_none = st.toggle("STRONG yoksa en yakın 10 adayı göster", value=True)
    show_cooldown_min = st.slider("Cooldown (dk)", 0, 240, 30, step=5)

    st.write("---")
    st.subheader("PRO Kapılar")
    pro_mode = st.toggle("PRO Mode", value=DEFAULT_PRO_MODE)

    require_qv = st.toggle("24h QuoteVolume filtresi", value=DEFAULT_REQUIRE_24H_QUOTEVOL, disabled=not pro_mode)
    min_qv = st.number_input("Min QuoteVolume (USDT)", value=int(DEFAULT_MIN_QUOTEVOL_USDT), step=500_000, disabled=not pro_mode)

    require_spread = st.toggle("Spread filtresi", value=DEFAULT_REQUIRE_SPREAD, disabled=not pro_mode)
    max_spread = st.slider("Max Spread (%)", 0.10, 1.50, float(DEFAULT_MAX_SPREAD_PCT), step=0.05, disabled=not pro_mode)

    require_regime_1h = st.toggle("1h EMA50/EMA200 rejim", value=DEFAULT_REQUIRE_REGIME_1H, disabled=not pro_mode)
    require_regime_4h = st.toggle("4h EMA50/EMA200 rejim", value=DEFAULT_REQUIRE_REGIME_4H, disabled=not pro_mode)

    require_adx_1h = st.toggle("1h ADX kapısı", value=DEFAULT_REQUIRE_ADX_1H, disabled=not pro_mode)
    adx1_min = st.slider("1h ADX min", 10.0, 40.0, 25.0, step=0.5, disabled=not pro_mode)

    require_adx_4h = st.toggle("4h ADX kapısı", value=DEFAULT_REQUIRE_ADX_4H, disabled=not pro_mode)
    adx4_min = st.slider("4h ADX min", 10.0, 40.0, 22.0, step=0.5, disabled=not pro_mode)

    require_vol = st.toggle("Hacim teyidi", value=DEFAULT_REQUIRE_VOL_CONFIRM, disabled=not pro_mode)
    vol_mult = st.slider("Vol çarpanı (Vol >= SMA20*x)", 1.0, 2.0, float(DEFAULT_VOL_MULT), step=0.05, disabled=not pro_mode)

    require_2c = st.toggle("2 Mum teyidi", value=DEFAULT_REQUIRE_2CANDLE_CONFIRM, disabled=not pro_mode)

    st.write("---")
    st.subheader("Otomatik Tarama")
    auto_scan = st.toggle("Auto Scan", value=DEFAULT_AUTO_SCAN)
    refresh_sec = st.slider("Tarama aralığı (sn)", 60, 1800, DEFAULT_REFRESH_SEC, step=60)
    manual_scan = st.button("🚀 Hemen Tara", use_container_width=True)


def try_autorefresh(interval_ms: int, key: str):
    try:
        return st.autorefresh(interval=interval_ms, key=key)
    except Exception:
        try:
            return st.experimental_autorefresh(interval=interval_ms, key=key)
        except Exception:
            return None


if auto_scan:
    try_autorefresh(interval_ms=int(refresh_sec * 1000), key="hardmode_refresh")


# Decide scan
do_scan = False
if AUTO_SCAN_ON_LOAD and not st.session_state["boot_scanned"]:
    do_scan = True
elif manual_scan:
    do_scan = True
elif auto_scan:
    last = st.session_state.get("last_scan_time")
    if last is None or (datetime.now(IST_TZ) - last >= timedelta(seconds=refresh_sec)):
        do_scan = True


# Run scan
if do_scan:
    ex = make_exchange()
    all_syms = load_usdt_spot_symbols()

    tickers = safe_fetch_tickers_all(ex) if pro_mode and require_qv else {}

    universe = all_syms[: min(len(all_syms), hard_max)]

    # QuoteVolume pre-filter
    qv_map: dict[str, float] = {}
    if pro_mode and require_qv and tickers:
        keep = []
        for s in universe:
            t = tickers.get(s)
            qv = quote_volume_usdt(t) if isinstance(t, dict) else 0.0
            qv_map[s] = float(qv)
            if qv >= float(min_qv):
                keep.append(s)
        universe = keep
    else:
        for s in universe:
            t = tickers.get(s) if isinstance(tickers, dict) else None
            qv_map[s] = quote_volume_usdt(t) if isinstance(t, dict) else 0.0

    progress = st.progress(0, text="Taranıyor…")
    status = st.empty()

    rows = []
    total = len(universe)

    # Phase 1: base scan
    for i, symbol in enumerate(universe, start=1):
        progress.progress(int((i - 1) / max(1, total) * 100), text=f"{base_tf} -> {symbol} ({i}/{total})")
        try:
            o = safe_fetch_ohlcv(ex, symbol, base_tf, base_limit)
            if not o:
                continue
            r = build_base_row(symbol, o, base_tf, require_2candle_confirm=(pro_mode and require_2c))
            if r is None:
                continue
            r["QV_24H"] = float(qv_map.get(symbol, 0.0))
            rows.append(r)
        except (ccxt.RequestTimeout, ccxt.NetworkError):
            status.warning(f"Timeout: {symbol}")
        except Exception:
            pass
        time.sleep(0.02)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Phase 2: only STRONG candidates + PRO gates
    strong_df = pd.DataFrame()
    if not df.empty:
        cand = df[(df["RAW"] >= STRONG_LONG_MIN) | (df["RAW"] <= STRONG_SHORT_MAX)].copy()

        if pro_mode and require_vol:
            cand = cand[(cand["VOL_SMA20"] > 0) & (cand["VOL_LAST"] >= cand["VOL_SMA20"] * float(vol_mult))].copy()

        if pro_mode and require_2c:
            cand = cand[cand["CONFIRM_OK"] == True].copy()

        if not cand.empty and pro_mode:
            kept_symbols = []
            info_map = {}

            c_total = len(cand)
            for j, row in enumerate(cand.itertuples(index=False), start=1):
                progress.progress(int((j - 1) / max(1, c_total) * 100), text=f"PRO kapılar (1h/4h) {j}/{c_total}")
                symbol = getattr(row, "SYMBOL")
                raw = int(getattr(row, "RAW"))
                want_dir = "LONG" if raw >= 50 else "SHORT"

                ok, info = pro_pass(
                    ex=ex,
                    symbol=symbol,
                    want_dir=want_dir,
                    require_regime_1h=require_regime_1h,
                    require_regime_4h=require_regime_4h,
                    require_adx_1h=require_adx_1h,
                    require_adx_4h=require_adx_4h,
                    adx1_min=float(adx1_min),
                    adx4_min=float(adx4_min),
                    require_spread=require_spread,
                    max_spread_pct=float(max_spread),
                )
                if ok:
                    kept_symbols.append(symbol)
                    info_map[symbol] = info
                time.sleep(0.02)

            if kept_symbols:
                strong_df = cand[cand["SYMBOL"].isin(set(kept_symbols))].copy()
                strong_df["SPREAD%"] = strong_df["SYMBOL"].map(lambda s: info_map.get(s, {}).get("SPREAD%", np.nan))
                strong_df["ADX_1H"] = strong_df["SYMBOL"].map(lambda s: info_map.get(s, {}).get("ADX_1H", np.nan))
                strong_df["ADX_4H"] = strong_df["SYMBOL"].map(lambda s: info_map.get(s, {}).get("ADX_4H", np.nan))
            else:
                strong_df = pd.DataFrame()
        else:
            strong_df = cand.copy()

    progress.progress(100, text="Tarama bitti.")
    status.empty()

    st.session_state["results"] = {"all": df, "strong": strong_df}
    st.session_state["last_scan_time"] = datetime.now(IST_TZ)
    st.session_state["boot_scanned"] = True


# =============================
# DISPLAY (SNIPER TABLE DIRECT)
# =============================
res = st.session_state.get("results")
last_scan = st.session_state.get("last_scan_time")
son = last_scan.strftime("%H:%M:%S") if last_scan else "—"
st.markdown(f"**🕒 Son Tarama:** `{son}`  •  **TF:** `{base_tf}`")


def apply_dir_filter(d: pd.DataFrame, f: str) -> pd.DataFrame:
    if f == "LONG":
        return d[d["YÖN"] == "LONG"].copy()
    if f == "SHORT":
        return d[d["YÖN"] == "SHORT"].copy()
    return d.copy()


st.subheader("🎯 SNIPER TABLO (Direkt Ekranda)")

if not res or not isinstance(res, dict):
    st.info("İlk tarama başlatılıyor…")
else:
    df_strong = res.get("strong")
    df_all = res.get("all")

    if isinstance(df_strong, pd.DataFrame) and not df_strong.empty:
        out = df_strong.copy()
        out = apply_dir_filter(out, dir_filter)
        out = out.sort_values(["SKOR", "RAW"], ascending=[False, False]).reset_index(drop=True)

        # Cooldown + take top N
        kept = []
        for _, r in out.iterrows():
            sym = str(r["SYMBOL"])
            direction = str(r["YÖN"])
            if cooldown_allow(sym, direction, int(show_cooldown_min)):
                kept.append(r)
                if len(kept) >= int(sniper_top):
                    break

        if kept:
            show = pd.DataFrame(kept)

            # mark cooldown for shown
            for _, r in show.iterrows():
                cooldown_mark(str(r["SYMBOL"]), str(r["YÖN"]))

            cols = ["COIN", "YÖN", "SKOR", "FİYAT", "RAW", "QV_24H"]
            for extra in ["SPREAD%", "ADX_1H", "ADX_4H"]:
                if extra in show.columns:
                    cols.append(extra)

            show = show.loc[:, [c for c in cols if c in show.columns]]
            st.dataframe(style_table(show), use_container_width=True, height=600)
        else:
            st.warning("STRONG var ama cooldown / yön filtresi yüzünden şu an tablo boş.")
    else:
        st.warning("⚠️ Şu an STRONG (RAW>=90 / RAW<=10) yok.")
        if show_near_if_none and isinstance(df_all, pd.DataFrame) and not df_all.empty:
            near = apply_dir_filter(df_all.copy(), dir_filter)
            near = near.sort_values(["SKOR", "RAW"], ascending=[False, False]).head(10).reset_index(drop=True)
            cols = ["COIN", "YÖN", "SKOR", "FİYAT", "RAW", "QV_24H"]
            near = near.loc[:, [c for c in cols if c in near.columns]]
            st.caption("STRONG değil ama en yakın 10 adayı gösteriyorum (kalabalık yok).")
            st.dataframe(style_table(near), use_container_width=True, height=520)
