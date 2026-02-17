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
EMA_REGIME_PERIOD = 200
ADX_PERIOD = 14
VOL_SMA_PERIOD = 20

# Strong signal gates (RAW score)
STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10

# When no strong signals exist
FALLBACK_TOP_N = 10  # show only closest candidates

# Universe size (KuCoin has many USDT pairs; keep cloud stable)
DEFAULT_HARD_MAX_SYMBOLS = 600
DEFAULT_BASE_LIMIT = 200

# Default PRO settings (hard/strict)
DEFAULT_PRO_MODE = True
DEFAULT_REQUIRE_REGIME_1H = True
DEFAULT_REQUIRE_REGIME_4H = True
DEFAULT_REQUIRE_ADX_1H = True
DEFAULT_REQUIRE_ADX_4H = True
DEFAULT_REQUIRE_VOL_CONFIRM = True
DEFAULT_REQUIRE_SPREAD = True

# Spread filter (orderbook)
DEFAULT_MAX_SPREAD_PCT = 0.60  # 0.60% is strict; adjust if too few signals

# Auto-scan behavior
AUTO_SCAN_ON_LOAD = True


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
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
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
# Score engine (unchanged style: 50 +/- steps)
# =============================
def raw_sniper_score(close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float) -> int:
    score = 50

    # Trend Filter (20)
    score += 20 if close > sma20_v else -20

    # Momentum Filter (40)
    if rsi_v < 35:
        score += 40
    elif rsi_v > 65:
        score -= 40

    # Volatility Filter (40)
    if close <= bb_low:
        score += 40
    elif close >= bb_up:
        score -= 40

    return int(max(0, min(100, score)))


def direction_from_raw(raw_score: int) -> str:
    return "LONG" if raw_score >= 50 else "SHORT"


def strength_from_raw(raw_score: int) -> int:
    # unified 0..100 strength:
    # LONG strength = raw
    # SHORT strength = 100 - raw
    return int(raw_score) if raw_score >= 50 else int(100 - raw_score)


def label_from_raw(raw_score: int) -> str:
    if raw_score >= STRONG_LONG_MIN:
        return "🔥 STRONG LONG"
    if raw_score <= STRONG_SHORT_MAX:
        return "💀 STRONG SHORT"
    return "⏳ WATCH"


# =============================
# KuCoin / CCXT helpers
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
        syms.append(sym)
    return sorted(set(syms))


def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def safe_fetch_orderbook_spread_pct(ex: ccxt.Exchange, symbol: str) -> float | None:
    """
    Returns spread% = (ask-bid)/mid * 100 if available, else None.
    """
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
# Build base row (base TF)
# =============================
def build_base_row(symbol: str, ohlcv: list, base_tf: str) -> dict | None:
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    if len(df) < max(SMA_PERIOD, BB_PERIOD, RSI_PERIOD, VOL_SMA_PERIOD) + 5:
        return None

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    sma20_s = sma(close, SMA_PERIOD)
    _, bb_up_s, bb_low_s = bollinger_bands(close, BB_PERIOD, BB_STD)
    rsi_s = rsi_wilder(close, RSI_PERIOD)
    vol_sma20 = sma(volume, VOL_SMA_PERIOD)

    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2])
    chg_pct = ((last_close - prev_close) / prev_close) * 100.0 if prev_close else 0.0

    last_sma20 = float(sma20_s.iloc[-1])
    last_rsi = float(rsi_s.iloc[-1])
    last_low = float(bb_low_s.iloc[-1])
    last_up = float(bb_up_s.iloc[-1])

    last_vol = float(volume.iloc[-1])
    last_vol_sma = float(vol_sma20.iloc[-1]) if not np.isnan(vol_sma20.iloc[-1]) else 0.0
    vol_ok = (last_vol_sma > 0) and (last_vol >= last_vol_sma * 1.2)

    if any(np.isnan([last_sma20, last_rsi, last_low, last_up])):
        return None

    raw = raw_sniper_score(last_close, last_sma20, last_rsi, last_low, last_up)
    yon = direction_from_raw(raw)
    skor = strength_from_raw(raw)
    etiket = label_from_raw(raw)

    coin = symbol.replace("/USDT", "")
    return {
        "SYMBOL": symbol,
        "COIN": coin,
        "YÖN": yon,
        "RAW": int(raw),
        "SKOR": int(skor),
        "FİYAT": float(last_close),
        "Change%": float(chg_pct),
        "VOL_OK": bool(vol_ok),
        "ETİKET": etiket,
        "BASE_TF": base_tf,
    }


# =============================
# PRO / Strict filters (1h & 4h)
# =============================
def strict_pass(
    ex: ccxt.Exchange,
    symbol: str,
    want_dir: str,
    require_regime_1h: bool,
    require_regime_4h: bool,
    require_adx_1h: bool,
    require_adx_4h: bool,
    require_spread: bool,
    max_spread_pct: float,
) -> tuple[bool, dict]:
    info = {
        "EMA200_1H": np.nan,
        "EMA200_4H": np.nan,
        "ADX_1H": np.nan,
        "ADX_4H": np.nan,
        "SPREAD%": np.nan,
        "REGIME1H_OK": True,
        "REGIME4H_OK": True,
        "ADX1H_OK": True,
        "ADX4H_OK": True,
        "SPREAD_OK": True,
    }

    # Spread gate first (fast reject)
    if require_spread:
        sp = safe_fetch_orderbook_spread_pct(ex, symbol)
        info["SPREAD%"] = sp if sp is not None else np.nan
        if sp is None or sp > max_spread_pct:
            info["SPREAD_OK"] = False
            return False, info

    # 1h gate
    if require_regime_1h or require_adx_1h:
        o1 = safe_fetch_ohlcv(ex, symbol, "1h", 260)
        if not o1 or len(o1) < max(EMA_REGIME_PERIOD, ADX_PERIOD) + 10:
            return False, info
        d1 = pd.DataFrame(o1, columns=["ts", "open", "high", "low", "close", "volume"])
        h1 = d1["high"].astype(float)
        l1 = d1["low"].astype(float)
        c1 = d1["close"].astype(float)

        ema200_1h = ema(c1, EMA_REGIME_PERIOD)
        last_close_1h = float(c1.iloc[-1])
        last_ema_1h = float(ema200_1h.iloc[-1]) if not np.isnan(ema200_1h.iloc[-1]) else np.nan
        info["EMA200_1H"] = last_ema_1h

        if require_regime_1h:
            if np.isnan(last_ema_1h):
                info["REGIME1H_OK"] = False
                return False, info
            info["REGIME1H_OK"] = (last_close_1h > last_ema_1h) if want_dir == "LONG" else (last_close_1h < last_ema_1h)
            if not info["REGIME1H_OK"]:
                return False, info

        if require_adx_1h:
            adx1 = adx_wilder(h1, l1, c1, ADX_PERIOD)
            last_adx1 = float(adx1.iloc[-1]) if not np.isnan(adx1.iloc[-1]) else 0.0
            info["ADX_1H"] = last_adx1
            info["ADX1H_OK"] = last_adx1 >= 20.0
            if not info["ADX1H_OK"]:
                return False, info

    # 4h gate
    if require_regime_4h or require_adx_4h:
        o4 = safe_fetch_ohlcv(ex, symbol, "4h", 260)
        if not o4 or len(o4) < max(EMA_REGIME_PERIOD, ADX_PERIOD) + 10:
            return False, info
        d4 = pd.DataFrame(o4, columns=["ts", "open", "high", "low", "close", "volume"])
        h4 = d4["high"].astype(float)
        l4 = d4["low"].astype(float)
        c4 = d4["close"].astype(float)

        ema200_4h = ema(c4, EMA_REGIME_PERIOD)
        last_close_4h = float(c4.iloc[-1])
        last_ema_4h = float(ema200_4h.iloc[-1]) if not np.isnan(ema200_4h.iloc[-1]) else np.nan
        info["EMA200_4H"] = last_ema_4h

        if require_regime_4h:
            if np.isnan(last_ema_4h):
                info["REGIME4H_OK"] = False
                return False, info
            info["REGIME4H_OK"] = (last_close_4h > last_ema_4h) if want_dir == "LONG" else (last_close_4h < last_ema_4h)
            if not info["REGIME4H_OK"]:
                return False, info

        if require_adx_4h:
            adx4 = adx_wilder(h4, l4, c4, ADX_PERIOD)
            last_adx4 = float(adx4.iloc[-1]) if not np.isnan(adx4.iloc[-1]) else 0.0
            info["ADX_4H"] = last_adx4
            info["ADX4H_OK"] = last_adx4 >= 18.0
            if not info["ADX4H_OK"]:
                return False, info

    return True, info


# =============================
# Streamlit Styling
# =============================
def style_strong_table(df: pd.DataFrame):
    def dir_style(v):
        s = str(v)
        if s == "LONG":
            return "background-color: #064e3b; color: #ffffff; font-weight: 700;"
        if s == "SHORT":
            return "background-color: #7f1d1d; color: #ffffff; font-weight: 700;"
        return ""

    def skor_style(v):
        try:
            x = float(v)
        except Exception:
            return ""
        if x >= 95:
            return "background-color: #006400; color: #ffffff; font-weight: 800;"
        if x >= 90:
            return "background-color: #0f3d0f; color: #ffffff; font-weight: 700;"
        return "background-color: #111827; color: #e5e7eb;"

    fmt = {"FİYAT": "{:.4f}", "SKOR": "{:.0f}", "RAW": "{:.0f}"}

    return (
        df.style.format(fmt)
        .applymap(dir_style, subset=["YÖN"])
        .applymap(skor_style, subset=["SKOR"])
        .set_properties(**{"border-color": "#1f2a37"})
    )


# =============================
# App UI
# =============================
st.set_page_config(page_title="KuCoin PRO Sniper (Spot)", layout="wide")
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

# Session state
if "results" not in st.session_state:
    st.session_state["results"] = None
if "last_scan_time" not in st.session_state:
    st.session_state["last_scan_time"] = None
if "boot_scanned" not in st.session_state:
    st.session_state["boot_scanned"] = False

# Header
c1, c2 = st.columns([2, 1], vertical_alignment="center")
with c1:
    st.title("KuCoin PRO Sniper — SERT Sinyal Tablosu (Kalabalık Yok)")
    st.caption("Sadece en güçlü LONG/SHORT sinyaller (RAW>=90 / RAW<=10) + PRO kapılar (EMA200 1h+4h, ADX 1h+4h, Hacim, Spread).")
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
    st.subheader("Ayarlar")

    base_tf = st.selectbox("Tarama TF", ["5m", "15m", "30m", "1h"], index=1)
    base_limit = st.slider("Mum sayısı", 120, 500, DEFAULT_BASE_LIMIT, step=20)
    hard_max = st.slider("Evren (USDT spot) kaç coin taransın", 100, 1200, DEFAULT_HARD_MAX_SYMBOLS, step=50)

    st.write("---")
    st.subheader("PRO Mod (Önerilir)")
    pro_mode = st.toggle("PRO Mode", value=DEFAULT_PRO_MODE)

    require_regime_1h = st.toggle("1h EMA200 Rejim", value=DEFAULT_REQUIRE_REGIME_1H, disabled=not pro_mode)
    require_regime_4h = st.toggle("4h EMA200 Rejim", value=DEFAULT_REQUIRE_REGIME_4H, disabled=not pro_mode)
    require_adx_1h = st.toggle("1h ADX >= 20", value=DEFAULT_REQUIRE_ADX_1H, disabled=not pro_mode)
    require_adx_4h = st.toggle("4h ADX >= 18", value=DEFAULT_REQUIRE_ADX_4H, disabled=not pro_mode)
    require_vol = st.toggle("Hacim Onayı (Vol > SMA20*1.2)", value=DEFAULT_REQUIRE_VOL_CONFIRM, disabled=not pro_mode)

    st.write("---")
    st.subheader("Spread Filtresi (ÇOK Önemli)")
    require_spread = st.toggle("Orderbook Spread filtresi", value=DEFAULT_REQUIRE_SPREAD, disabled=not pro_mode)
    max_spread = st.slider("Max Spread (%)", 0.10, 2.00, float(DEFAULT_MAX_SPREAD_PCT), step=0.05, disabled=not pro_mode)

    st.write("---")
    auto_scan = st.toggle("Auto Scan", value=True)
    refresh_sec = st.slider("Tarama aralığı (sn)", 60, 1800, 240, step=60)
    manual_scan = st.button("🚀 Hemen Tara", use_container_width=True)

# Autorefresh
def try_autorefresh(interval_ms: int, key: str):
    try:
        return st.autorefresh(interval=interval_ms, key=key)
    except Exception:
        try:
            return st.experimental_autorefresh(interval=interval_ms, key=key)
        except Exception:
            return None

if auto_scan:
    try_autorefresh(interval_ms=int(refresh_sec * 1000), key="pro_sniper_refresh")

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
    syms = load_usdt_spot_symbols()
    universe = syms[: min(len(syms), hard_max)]

    progress = st.progress(0, text="Taranıyor…")
    status = st.empty()

    rows = []
    total = len(universe)

    # Phase 1: base TF scan
    for i, symbol in enumerate(universe, start=1):
        progress.progress(int((i - 1) / max(1, total) * 100), text=f"{base_tf} -> {symbol} ({i}/{total})")
        try:
            o = safe_fetch_ohlcv(ex, symbol, base_tf, base_limit)
            if not o:
                continue
            r = build_base_row(symbol, o, base_tf)
            if r is None:
                continue
            rows.append(r)
        except (ccxt.RequestTimeout, ccxt.NetworkError):
            status.warning(f"Timeout: {symbol}")
        except Exception:
            pass
        time.sleep(0.02)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Phase 2: strong candidates only + PRO gates
    strong_df = pd.DataFrame()
    if not df.empty:
        cand = df[(df["RAW"] >= STRONG_LONG_MIN) | (df["RAW"] <= STRONG_SHORT_MAX)].copy()

        # volume gate (base tf)
        if pro_mode and require_vol:
            cand = cand[cand["VOL_OK"] == True].copy()

        if not cand.empty and pro_mode:
            kept = []
            c_total = len(cand)
            for j, row in enumerate(cand.itertuples(index=False), start=1):
                progress.progress(int((j - 1) / max(1, c_total) * 100), text=f"PRO kapılar (1h/4h) {j}/{c_total}")
                symbol = getattr(row, "SYMBOL")
                raw = int(getattr(row, "RAW"))
                want_dir = "LONG" if raw >= 50 else "SHORT"
                ok, info = strict_pass(
                    ex=ex,
                    symbol=symbol,
                    want_dir=want_dir,
                    require_regime_1h=require_regime_1h,
                    require_regime_4h=require_regime_4h,
                    require_adx_1h=require_adx_1h,
                    require_adx_4h=require_adx_4h,
                    require_spread=require_spread,
                    max_spread_pct=max_spread,
                )
                if ok:
                    kept.append((symbol, info))
                time.sleep(0.02)

            if kept:
                info_map = {s: inf for s, inf in kept}
                keep_set = set(info_map.keys())
                strong_df = cand[cand["SYMBOL"].isin(keep_set)].copy()
                strong_df["SPREAD%"] = strong_df["SYMBOL"].map(lambda s: info_map.get(s, {}).get("SPREAD%", np.nan))
                strong_df["ADX_1H"] = strong_df["SYMBOL"].map(lambda s: info_map.get(s, {}).get("ADX_1H", np.nan))
                strong_df["ADX_4H"] = strong_df["SYMBOL"].map(lambda s: info_map.get(s, {}).get("ADX_4H", np.nan))
            else:
                strong_df = pd.DataFrame()
        else:
            # pro_mode off: strong signals are only raw gates (+ optional vol if on)
            strong_df = cand.copy()

    progress.progress(100, text="Tarama bitti.")
    status.empty()

    st.session_state["results"] = {"all": df, "strong": strong_df}
    st.session_state["last_scan_time"] = datetime.now(IST_TZ)
    st.session_state["boot_scanned"] = True

# Show results
res = st.session_state.get("results")
last_scan = st.session_state.get("last_scan_time")
son = last_scan.strftime("%H:%M:%S") if last_scan else "—"
st.markdown(f"**🕒 Son Tarama:** `{son}`  •  **TF:** `{base_tf}`")

if not res or not isinstance(res, dict):
    st.info("İlk tarama başlatılıyor…")
else:
    df_all = res.get("all")
    df_strong = res.get("strong")

    # Strong table first (the one you want)
    if isinstance(df_strong, pd.DataFrame) and not df_strong.empty:
        show_cols = ["COIN", "YÖN", "SKOR", "FİYAT", "RAW", "ETİKET"]
        # add PRO diagnostics if present
        if "SPREAD%" in df_strong.columns:
            show_cols += ["SPREAD%"]
        if "ADX_1H" in df_strong.columns:
            show_cols += ["ADX_1H"]
        if "ADX_4H" in df_strong.columns:
            show_cols += ["ADX_4H"]

        out = df_strong.copy()
        out = out.sort_values(["SKOR", "RAW"], ascending=[False, False]).reset_index(drop=True)
        out = out.loc[:, [c for c in show_cols if c in out.columns]]

        # Keep it NOT crowded
        st.subheader("✅ EN GÜÇLÜ SİNYALLER (SERT)")
        st.dataframe(style_strong_table(out), use_container_width=True, height=520)

    else:
        st.subheader("⚠️ Şu an STRONG (RAW>=90 / RAW<=10) yok")
        st.caption("Kalabalık yapmıyorum. En yakın adaylardan sadece ilk 10’u gösteriyorum (STRONG DEĞİL).")

        if isinstance(df_all, pd.DataFrame) and not df_all.empty:
            # proximity to strong:
            # for LONG side: closeness = RAW (higher is closer to 90)
            # for SHORT side: closeness = (100-RAW) (higher is closer to 10)
            tmp = df_all.copy()
            tmp["YAKINLIK"] = tmp["SKOR"]  # already unified strength
            tmp = tmp.sort_values("YAKINLIK", ascending=False).head(FALLBACK_TOP_N).reset_index(drop=True)

            fallback = tmp[["COIN", "YÖN", "SKOR", "FİYAT", "RAW"]].copy()
            st.dataframe(style_strong_table(fallback), use_container_width=True, height=420)
        else:
            st.info("Veri gelmedi. Evren/tf ayarını değiştirip tekrar tara.")
