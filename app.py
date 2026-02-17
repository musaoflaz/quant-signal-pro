# requirements.txt (install these):
# streamlit
# pandas
# numpy
# ccxt
#
# SECURITY NOTE:
# - Do NOT hardcode Telegram token in this file.
# - Prefer Streamlit Cloud -> App settings -> Secrets:
#   TELEGRAM_BOT_TOKEN="xxxx"
#   TELEGRAM_CHAT_ID="1358384022"
# - If you ever shared a token anywhere, rotate it in @BotFather.

from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib import request, parse

import numpy as np
import pandas as pd
import streamlit as st
import ccxt


# -----------------------------
# Config
# -----------------------------
IST_TZ = ZoneInfo("Europe/Istanbul")

RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIOD = 20

EMA_REGIME_PERIOD = 200  # 1h EMA200 regime filter
ADX_PERIOD = 14          # 1h ADX filter
VOL_SMA_PERIOD = 20      # volume confirmation on base TF

# Raw score gates (unchanged)
STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10

# What you want on main screen
TOP_SHOW = 20

# Auto scan without clicking (first load)
AUTO_SCAN_ON_LOAD = True

# Cloud safety
DEFAULT_HARD_MAX_SYMBOLS = 500  # keep stable on Streamlit Cloud
DEFAULT_CANDLE_LIMIT = 200

# Strict filters default (quality > quantity)
DEFAULT_STRICT_MODE = True
DEFAULT_REQUIRE_REGIME = True
DEFAULT_REQUIRE_ADX = True
DEFAULT_REQUIRE_VOL_CONFIRM = True

# Telegram alert policy (only strong signals)
ALERT_LONG_RAW_MIN = 90
ALERT_SHORT_RAW_MAX = 10


# -----------------------------
# Indicators (pure pandas/numpy)
# -----------------------------
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
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)

    # Wilder smoothing (EMA with alpha=1/period)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_dm_s = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_s = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (plus_dm_s / atr.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm_s / atr.replace(0.0, np.nan))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)


# -----------------------------
# Scoring (unchanged logic)
# -----------------------------
def raw_sniper_score(close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float) -> int:
    score = 50

    # Trend Filter (20)
    if close > sma20_v:
        score += 20
    else:
        score -= 20

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


def confidence_from_raw(raw_score: int) -> int:
    # Unified 0..100 strength for BOTH directions:
    # LONG: raw=90 -> 90
    # SHORT: raw=10 -> 90
    if raw_score >= 50:
        return int(raw_score)
    return int(100 - raw_score)


def signal_label(raw_score: int) -> str:
    if raw_score >= STRONG_LONG_MIN:
        return "🔥 STRONG LONG SIGNAL"
    if raw_score <= STRONG_SHORT_MAX:
        return "💀 STRONG SHORT SIGNAL"
    return "⏳ WATCHING"


# -----------------------------
# KuCoin / CCXT
# -----------------------------
def make_exchange() -> ccxt.kucoin:
    return ccxt.kucoin(
        {
            "enableRateLimit": True,
            "timeout": 20000,  # ms
        }
    )


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


# -----------------------------
# Telegram (stdlib only)
# -----------------------------
def telegram_send_message(token: str, chat_id: str, text: str, timeout: int = 12) -> tuple[bool, str]:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        form = parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
        req = request.Request(
            url=url,
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "kucoin-quant-sniper"},
            method="POST",
        )
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        return True, body[:200]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def build_alert_message(row: pd.Series, timeframe: str) -> str:
    return (
        f"{row['SIGNAL']}\n"
        f"COIN: {row['COIN']}/USDT\n"
        f"YÖN: {row['YÖN']}\n"
        f"SKOR: {int(row['SKOR'])}\n"
        f"FİYAT: {row['FİYAT']:.4f}\n"
        f"TF: {timeframe}\n"
        f"Time(IST): {datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
    )


def should_alert(alert_state: dict, coin: str, signal: str, cooldown_minutes: int) -> bool:
    key = f"{coin}|{signal}"
    last_ts = alert_state.get(key)
    if not last_ts:
        return True
    try:
        last_dt = datetime.fromisoformat(last_ts)
    except Exception:
        return True
    return datetime.now(IST_TZ) - last_dt >= timedelta(minutes=cooldown_minutes)


def mark_alert(alert_state: dict, coin: str, signal: str):
    key = f"{coin}|{signal}"
    alert_state[key] = datetime.now(IST_TZ).isoformat()


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="KuCoin Quant Sniper (Spot)", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.0rem; }
h1, h2, h3, h4, h5, h6, p, span, div { color: #e6edf3; }
[data-testid="stSidebar"] { background-color: #0b0f14; border-right: 1px solid #1f2a37; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.stButton>button {
    border: 1px solid #2d3b4d;
    background: #111827;
    color: #e6edf3;
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
}
.stButton>button:hover { border-color: #4b5563; background: #0f172a; }
</style>
""",
    unsafe_allow_html=True,
)

# Session state
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "last_scan_time" not in st.session_state:
    st.session_state["last_scan_time"] = None
if "last_scan_tf" not in st.session_state:
    st.session_state["last_scan_tf"] = None
if "alert_state" not in st.session_state:
    st.session_state["alert_state"] = {}
if "boot_scanned" not in st.session_state:
    st.session_state["boot_scanned"] = False
if "paused" not in st.session_state:
    st.session_state["paused"] = False

# Telegram defaults from secrets (preferred)
secret_token = ""
secret_chat = "1358384022"
try:
    secret_token = str(st.secrets.get("TELEGRAM_BOT_TOKEN", "")).strip()
    secret_chat = str(st.secrets.get("TELEGRAM_CHAT_ID", "1358384022")).strip()
except Exception:
    pass

if "tg_token" not in st.session_state:
    st.session_state["tg_token"] = secret_token
if "tg_chat_id" not in st.session_state:
    st.session_state["tg_chat_id"] = secret_chat


def try_autorefresh(interval_ms: int, key: str):
    try:
        return st.autorefresh(interval=interval_ms, key=key)
    except Exception:
        try:
            return st.experimental_autorefresh(interval=interval_ms, key=key)
        except Exception:
            return None


def style_main_table(df: pd.DataFrame):
    def score_bg(v):
        try:
            v = float(v)
        except Exception:
            return ""
        if v >= 90:
            return "background-color: #006400; color: #ffffff;"
        if v <= 60:
            return "background-color: #1f2937; color: #e5e7eb;"
        return "background-color: #0f172a; color: #ffffff;"

    def dir_bg(v):
        s = str(v)
        if "LONG" in s:
            return "background-color: #064e3b; color: #ffffff;"
        if "SHORT" in s:
            return "background-color: #7f1d1d; color: #ffffff;"
        return ""

    fmt = {"FİYAT": "{:.4f}", "SKOR": "{:.0f}"}

    return (
        df.style.format(fmt)
        .applymap(score_bg, subset=["SKOR"])
        .applymap(dir_bg, subset=["YÖN"])
        .set_properties(**{"border-color": "#1f2a37"})
    )


# -----------------------------
# Core scan with strict filters
# -----------------------------
def build_row_from_base_tf(symbol: str, ohlcv: list, base_tf: str) -> dict | None:
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    if len(df) < max(SMA_PERIOD, BB_PERIOD, RSI_PERIOD, VOL_SMA_PERIOD) + 5:
        return None

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
    conf = confidence_from_raw(raw)
    yon = direction_from_raw(raw)
    sig = signal_label(raw)

    coin = symbol.replace("/USDT", "")

    return {
        "SYMBOL": symbol,
        "COIN": coin,
        "FİYAT": last_close,
        "Change%": chg_pct,
        "RSI14": last_rsi,
        "SMA20": last_sma20,
        "BB_Lower": last_low,
        "BB_Upper": last_up,
        "RAW": raw,
        "SKOR": conf,           # unified 0..100 strength
        "YÖN": f"🟢 {yon}" if yon == "LONG" else f"🔴 {yon}",
        "SIGNAL": sig,
        "VOL_OK": bool(vol_ok),
        "BASE_TF": base_tf,
    }


def strict_filters_pass(ex: ccxt.Exchange, symbol: str, want_dir: str, require_regime: bool, require_adx: bool) -> tuple[bool, dict]:
    """
    Strict filters on 1h:
    - Regime: 1h close vs EMA200
    - ADX: 1h ADX >= 18 (trend strength / stability)
    """
    info = {"REGIME_OK": True, "ADX_OK": True, "EMA200_1H": np.nan, "ADX_1H": np.nan}

    # Fetch 1h candles
    # Need EMA200 and ADX14 => ~250 candles is enough
    try:
        ohlcv_1h = safe_fetch_ohlcv(ex, symbol, "1h", 260)
    except Exception:
        return False, info

    if not ohlcv_1h or len(ohlcv_1h) < max(EMA_REGIME_PERIOD, ADX_PERIOD) + 10:
        return False, info

    d1 = pd.DataFrame(ohlcv_1h, columns=["ts", "open", "high", "low", "close", "volume"])
    h = d1["high"].astype(float)
    l = d1["low"].astype(float)
    c = d1["close"].astype(float)

    ema200 = ema(c, EMA_REGIME_PERIOD)
    last_close = float(c.iloc[-1])
    last_ema200 = float(ema200.iloc[-1]) if not np.isnan(ema200.iloc[-1]) else np.nan
    info["EMA200_1H"] = last_ema200

    if require_regime:
        if np.isnan(last_ema200):
            info["REGIME_OK"] = False
            return False, info
        if want_dir == "LONG":
            info["REGIME_OK"] = last_close > last_ema200
        else:
            info["REGIME_OK"] = last_close < last_ema200
        if not info["REGIME_OK"]:
            return False, info

    if require_adx:
        adx = adx_wilder(h, l, c, ADX_PERIOD)
        last_adx = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0
        info["ADX_1H"] = last_adx
        info["ADX_OK"] = last_adx >= 18.0
        if not info["ADX_OK"]:
            return False, info

    return True, info


def run_scan(
    base_tf: str,
    base_limit: int,
    hard_max_symbols: int,
    strict_mode: bool,
    require_regime: bool,
    require_adx: bool,
    require_vol_confirm: bool,
) -> pd.DataFrame:
    ex = make_exchange()
    syms = load_usdt_spot_symbols()
    universe = syms[: min(len(syms), hard_max_symbols)]

    rows: list[dict] = []
    progress = st.progress(0, text="Tüm coinler taranıyor…")
    status = st.empty()

    total = len(universe)

    # 1) Base TF scan for all
    for i, symbol in enumerate(universe, start=1):
        progress.progress(int((i - 1) / total * 100), text=f"Base TF {base_tf} -> {symbol} ({i}/{total})")
        try:
            ohlcv = safe_fetch_ohlcv(ex, symbol, base_tf, base_limit)
            if not ohlcv:
                continue
            r = build_row_from_base_tf(symbol, ohlcv, base_tf)
            if r is None:
                continue
            rows.append(r)
        except (ccxt.RequestTimeout, ccxt.NetworkError):
            status.warning(f"Timeout/Network issue on {symbol}")
        except ccxt.ExchangeError:
            status.info(f"Exchange error on {symbol}")
        except Exception:
            pass
        time.sleep(0.02)

    if not rows:
        progress.progress(100, text="Tarama bitti.")
        status.empty()
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 2) If strict mode: only check 1h filters for potential strong candidates (cuts API calls)
    if strict_mode:
        # Candidate set: strong raw signals only
        cand = df[(df["RAW"] >= STRONG_LONG_MIN) | (df["RAW"] <= STRONG_SHORT_MAX)].copy()
        if cand.empty:
            progress.progress(100, text="Tarama bitti (strict: aday yok).")
            status.empty()
            return df.sort_values("SKOR", ascending=False).reset_index(drop=True)

        # Volume confirmation gate (on base TF) if requested
        if require_vol_confirm:
            cand = cand[cand["VOL_OK"] == True].copy()
            if cand.empty:
                progress.progress(100, text="Tarama bitti (strict: hacim onayı yok).")
                status.empty()
                return df.sort_values("SKOR", ascending=False).reset_index(drop=True)

        # Apply 1h strict filters
        kept = []
        c_total = len(cand)
        for j, row in enumerate(cand.itertuples(index=False), start=1):
            progress.progress(int((j - 1) / max(1, c_total) * 100), text=f"Strict 1h filter ({j}/{c_total})")
            symbol = getattr(row, "SYMBOL")
            raw = int(getattr(row, "RAW"))
            want_dir = "LONG" if raw >= 50 else "SHORT"
            try:
                ok, info = strict_filters_pass(ex, symbol, want_dir, require_regime, require_adx)
                if ok:
                    kept.append((symbol, info))
            except Exception:
                continue
            time.sleep(0.02)

        if kept:
            keep_set = {s for s, _ in kept}
            info_map = {s: info for s, info in kept}

            df["STRICT_OK"] = df["SYMBOL"].isin(keep_set)
            df["EMA200_1H"] = df["SYMBOL"].map(lambda s: info_map.get(s, {}).get("EMA200_1H", np.nan))
            df["ADX_1H"] = df["SYMBOL"].map(lambda s: info_map.get(s, {}).get("ADX_1H", np.nan))
        else:
            df["STRICT_OK"] = False
            df["EMA200_1H"] = np.nan
            df["ADX_1H"] = np.nan

    progress.progress(100, text="Tarama bitti.")
    status.empty()

    # Sort by unified SKOR (strength)
    df = df.sort_values("SKOR", ascending=False).reset_index(drop=True)
    return df


def maybe_send_alerts(df: pd.DataFrame, timeframe: str, enable_telegram: bool, tg_token: str, tg_chat_id: str, cooldown_minutes: int):
    if df is None or df.empty or not enable_telegram:
        return
    if not tg_token.strip() or not tg_chat_id.strip():
        st.warning("Telegram açık ama token/chat_id boş.")
        return

    alert_state = st.session_state["alert_state"]

    # Only strong signals
    strong = df[(df["RAW"] >= ALERT_LONG_RAW_MIN) | (df["RAW"] <= ALERT_SHORT_RAW_MAX)].copy()
    if strong.empty:
        return

    # If strict mode is ON and we computed STRICT_OK, only alert strict OK
    if "STRICT_OK" in strong.columns:
        strong = strong[(strong["STRICT_OK"] == True) | (strong["STRICT_OK"].isna())]

    if strong.empty:
        return

    errors = []
    sent_any = False

    for _, row in strong.iterrows():
        coin = str(row["COIN"])
        sig = str(row["SIGNAL"])
        if not should_alert(alert_state, coin, sig, cooldown_minutes):
            continue

        msg = build_alert_message(
            pd.Series(
                {
                    "SIGNAL": row["SIGNAL"],
                    "COIN": row["COIN"],
                    "YÖN": row["YÖN"],
                    "SKOR": row["SKOR"],
                    "FİYAT": row["FİYAT"],
                }
            ),
            timeframe,
        )
        ok, info = telegram_send_message(tg_token.strip(), tg_chat_id.strip(), msg)
        if not ok:
            errors.append(f"{coin}: {info}")
        mark_alert(alert_state, coin, sig)
        sent_any = True

    if sent_any:
        st.toast("✅ Telegram bildirim kontrolü tamamlandı.", icon="🔔")
    if errors:
        with st.expander("Telegram errors"):
            for e in errors[:30]:
                st.write(f"- {e}")


# -----------------------------
# Header
# -----------------------------
left, right = st.columns([2, 1], vertical_alignment="center")
with left:
    st.title("KuCoin Quant Sniper (Spot) — Top 20 Tablo (Tıklamadan)")
    st.caption("Skor: RSI(14) + Bollinger(20,2) + SMA(20) • Strict Mode: 1h EMA200 + 1h ADX + Hacim onayı")
with right:
    now_ist = datetime.now(IST_TZ)
    st.markdown(
        f"""
<div style="text-align:right; padding-top: 8px;">
  <div style="font-size: 12px; opacity: 0.85;">Istanbul Time</div>
  <div style="font-size: 18px; font-weight: 700;">{now_ist.strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Kontrol")

    base_tf = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h"], index=1)
    base_limit = st.slider("Candles (base TF)", 120, 500, DEFAULT_CANDLE_LIMIT, step=20)

    hard_max = st.slider("Kaç coin taransın (stabilite)", 100, 1200, DEFAULT_HARD_MAX_SYMBOLS, step=50)

    st.write("---")
    st.subheader("Sinyal Kalitesi (Strict Mode)")
    strict_mode = st.toggle("Strict Mode (Önerilir)", value=DEFAULT_STRICT_MODE)
    require_regime = st.toggle("1h EMA200 Rejim Filtresi", value=DEFAULT_REQUIRE_REGIME, disabled=not strict_mode)
    require_adx = st.toggle("1h ADX >= 18 Filtresi", value=DEFAULT_REQUIRE_ADX, disabled=not strict_mode)
    require_vol = st.toggle("Hacim Onayı (Vol > SMA20*1.2)", value=DEFAULT_REQUIRE_VOL_CONFIRM, disabled=not strict_mode)

    st.write("---")
    st.subheader("Otomatik Tarama")
    auto_scan = st.toggle("Auto Scan", value=True)
    refresh_sec = st.slider("Tarama aralığı (sn)", 60, 1800, 240, step=60)

    st.write("---")
    st.subheader("SİSTEMİ DURDUR")
    st.session_state["paused"] = st.toggle("Duraklat", value=st.session_state["paused"])

    st.write("---")
    st.subheader("Telegram (Opsiyonel)")
    enable_telegram = st.toggle("Enable Telegram", value=False)

    tg_token_in = st.text_input("Telegram Bot Token", value=st.session_state["tg_token"], type="password")
    tg_chat_in = st.text_input("Chat ID", value=st.session_state["tg_chat_id"])
    cooldown_minutes = st.slider("Cooldown (dk)", 1, 180, 30, step=1)
    test_btn = st.button("🔔 Test Notification", use_container_width=True)

    st.session_state["tg_token"] = tg_token_in
    st.session_state["tg_chat_id"] = tg_chat_in

    st.write("---")
    manual_scan = st.button("🚀 Hemen Tara", use_container_width=True)


# Auto refresh
if auto_scan and not st.session_state["paused"]:
    try_autorefresh(interval_ms=int(refresh_sec * 1000), key="kucoin_auto_refresh")


# Telegram test
if test_btn:
    if not st.session_state["tg_token"].strip():
        st.error("Token boş. Secrets veya input ile gir.")
    elif not st.session_state["tg_chat_id"].strip():
        st.error("Chat ID boş.")
    else:
        msg = f"✅ Test Notification\nTime(IST): {datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M:%S')}\nTF: {base_tf}"
        ok, info = telegram_send_message(st.session_state["tg_token"].strip(), st.session_state["tg_chat_id"].strip(), msg)
        if ok:
            st.success("Test mesajı Telegram’a gönderildi.")
        else:
            st.error(f"Test failed: {info}")


# Decide scan
do_scan = False
if AUTO_SCAN_ON_LOAD and not st.session_state["boot_scanned"] and not st.session_state["paused"]:
    do_scan = True
elif manual_scan and not st.session_state["paused"]:
    do_scan = True
elif auto_scan and not st.session_state["paused"]:
    last_scan = st.session_state.get("last_scan_time")
    if last_scan is None:
        do_scan = True
    else:
        if datetime.now(IST_TZ) - last_scan >= timedelta(seconds=refresh_sec):
            do_scan = True


# Run scan
if do_scan:
    try:
        df_out = run_scan(
            base_tf=base_tf,
            base_limit=base_limit,
            hard_max_symbols=hard_max,
            strict_mode=strict_mode,
            require_regime=require_regime,
            require_adx=require_adx,
            require_vol_confirm=require_vol,
        )
        st.session_state["results_df"] = df_out
        st.session_state["last_scan_time"] = datetime.now(IST_TZ)
        st.session_state["last_scan_tf"] = base_tf
        st.session_state["boot_scanned"] = True

        maybe_send_alerts(
            df=df_out,
            timeframe=base_tf,
            enable_telegram=enable_telegram,
            tg_token=st.session_state["tg_token"],
            tg_chat_id=st.session_state["tg_chat_id"],
            cooldown_minutes=cooldown_minutes,
        )
    except Exception as e:
        st.error(f"Scan failed: {type(e).__name__}: {e}")


# -----------------------------
# MAIN SCREEN TABLE (no clicking)
# -----------------------------
df_res = st.session_state.get("results_df")
son_tarama = st.session_state["last_scan_time"].strftime("%H:%M:%S") if st.session_state["last_scan_time"] else "—"
st.markdown(f"**🕒 Son Tarama:** `{son_tarama}`  •  **TF:** `{st.session_state.get('last_scan_tf') or base_tf}`")

if isinstance(df_res, pd.DataFrame) and not df_res.empty:
    # If strict mode computed STRICT_OK, prioritize strict-ok rows first
    if "STRICT_OK" in df_res.columns and strict_mode:
        df_rank = df_res.copy()
        df_rank["__prio"] = np.where(df_rank["STRICT_OK"] == True, 1, 0)
        df_rank = df_rank.sort_values(["__prio", "SKOR"], ascending=[False, False]).drop(columns=["__prio"])
    else:
        df_rank = df_res.sort_values("SKOR", ascending=False)

    top20 = df_rank.head(TOP_SHOW).copy()

    # The exact simple table you asked (COIN / YÖN / SKOR / FİYAT)
    main_table = pd.DataFrame(
        {
            "COIN": top20["COIN"],
            "YÖN": top20["YÖN"],
            "SKOR": top20["SKOR"].astype(int),
            "FİYAT": top20["FİYAT"].astype(float),
        }
    )

    # Show immediately
    st.dataframe(style_main_table(main_table), use_container_width=True, height=540)

    # Optional detail tabs (keep your existing structure, but main table already shows)
    with st.expander("Detay (opsiyonel)", expanded=False):
        cols = ["COIN", "SYMBOL", "YÖN", "SKOR", "RAW", "FİYAT", "Change%", "RSI14", "SMA20", "BB_Lower", "BB_Upper", "VOL_OK"]
        if "EMA200_1H" in df_res.columns:
            cols += ["EMA200_1H"]
        if "ADX_1H" in df_res.columns:
            cols += ["ADX_1H"]
        if "STRICT_OK" in df_res.columns:
            cols += ["STRICT_OK"]
        show = df_res[cols].copy()
        st.dataframe(show, use_container_width=True, height=520)

else:
    st.info("İlk tarama başlatılıyor… (tablo birazdan dolacak)")
