# requirements.txt (install these):
# streamlit
# pandas
# numpy
# ccxt

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

CANDLE_LIMIT = 200
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIOD = 20

TOP_N = 100

STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10


# -----------------------------
# Indicator math (pure pandas/numpy)
# -----------------------------
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


# -----------------------------
# KuCoin / CCXT helpers
# -----------------------------
def make_exchange() -> ccxt.kucoin:
    return ccxt.kucoin(
        {
            "enableRateLimit": True,
            "timeout": 20000,  # ms
        }
    )


@st.cache_data(show_spinner=False, ttl=300)
def load_usdt_spot_symbols() -> list[str]:
    ex = make_exchange()
    markets = ex.load_markets()
    syms = []
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


def safe_fetch_tickers(ex: ccxt.Exchange, symbols: list[str]) -> dict:
    try:
        return ex.fetch_tickers(symbols)
    except Exception:
        try:
            all_t = ex.fetch_tickers()
            return {s: all_t.get(s) for s in symbols if s in all_t}
        except Exception:
            return {}


def compute_liquidity_rank(ticker: dict) -> float:
    if not ticker or not isinstance(ticker, dict):
        return 0.0
    qv = ticker.get("quoteVolume")
    if qv is not None:
        try:
            return float(qv)
        except Exception:
            pass
    bv = ticker.get("baseVolume")
    last = ticker.get("last")
    try:
        if bv is not None and last is not None:
            return float(bv) * float(last)
    except Exception:
        pass
    return 0.0


def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def score_asset(close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float) -> int:
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


def signal_label(score: int) -> str:
    if score >= STRONG_LONG_MIN:
        return "🔥 STRONG LONG SIGNAL"
    if score <= STRONG_SHORT_MAX:
        return "💀 STRONG SHORT SIGNAL"
    return "⏳ WATCHING"


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
        f"{row['Signal']}\n"
        f"Symbol: {row['Symbol']}\n"
        f"TF: {timeframe}\n"
        f"Score: {int(row['Score'])}\n"
        f"Last: {row['Last']:.4f}\n"
        f"Change%: {row['Change%']:.2f}\n"
        f"RSI14: {row['RSI14']:.2f}\n"
        f"SMA20: {row['SMA20']:.4f}\n"
        f"BB: [{row['BB_Lower']:.4f} .. {row['BB_Upper']:.4f}]\n"
        f"Time(IST): {datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
    )


def should_alert(alert_state: dict, symbol: str, signal: str, cooldown_minutes: int) -> bool:
    key = f"{symbol}|{signal}"
    last_ts = alert_state.get(key)
    if not last_ts:
        return True
    try:
        last_dt = datetime.fromisoformat(last_ts)
    except Exception:
        return True
    return datetime.now(IST_TZ) - last_dt >= timedelta(minutes=cooldown_minutes)


def mark_alert(alert_state: dict, symbol: str, signal: str):
    key = f"{symbol}|{signal}"
    alert_state[key] = datetime.now(IST_TZ).isoformat()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="KuCoin Quant Sniper (Spot)", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.2rem; }
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
if "tg_token" not in st.session_state:
    st.session_state["tg_token"] = ""
if "tg_chat_id" not in st.session_state:
    st.session_state["tg_chat_id"] = "1358384022"


def try_autorefresh(interval_ms: int, key: str):
    try:
        return st.autorefresh(interval=interval_ms, key=key)
    except Exception:
        try:
            return st.experimental_autorefresh(interval=interval_ms, key=key)
        except Exception:
            return None


def style_scores(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def score_bg(v):
        try:
            v = float(v)
        except Exception:
            return ""
        if v >= STRONG_LONG_MIN:
            return "background-color: #006400; color: #ffffff;"
        if v <= STRONG_SHORT_MAX:
            return "background-color: #8B0000; color: #ffffff;"
        return ""

    fmt = {
        "Last": "{:.4f}",
        "Change%": "{:.2f}",
        "RSI14": "{:.2f}",
        "SMA20": "{:.4f}",
        "BB_Lower": "{:.4f}",
        "BB_Upper": "{:.4f}",
    }

    return (
        df.style.format(fmt)
        .applymap(score_bg, subset=["Score"])
        .set_properties(**{"border-color": "#1f2a37"})
    )


def run_scan(symbols: list[str], timeframe: str, limit: int) -> pd.DataFrame:
    ex = make_exchange()

    tickers = safe_fetch_tickers(ex, symbols)
    ranked = [(s, compute_liquidity_rank(tickers.get(s))) for s in symbols]
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [s for s, _ in ranked[:TOP_N]]

    rows = []
    progress = st.progress(0, text="Scanning markets…")
    status = st.empty()

    total = len(top_symbols)
    for i, symbol in enumerate(top_symbols, start=1):
        progress.progress(int((i - 1) / total * 100), text=f"Fetching {symbol} ({i}/{total})")
        try:
            ohlcv = safe_fetch_ohlcv(ex, symbol, timeframe, limit)
            if not ohlcv or len(ohlcv) < max(SMA_PERIOD, BB_PERIOD, RSI_PERIOD) + 5:
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            close = df["close"].astype(float)

            sma20_s = sma(close, SMA_PERIOD)
            _, bb_up_s, bb_low_s = bollinger_bands(close, BB_PERIOD, BB_STD)
            rsi_s = rsi_wilder(close, RSI_PERIOD)

            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            chg_pct = ((last_close - prev_close) / prev_close) * 100.0 if prev_close else 0.0

            last_sma20 = float(sma20_s.iloc[-1])
            last_rsi = float(rsi_s.iloc[-1])
            last_low = float(bb_low_s.iloc[-1])
            last_up = float(bb_up_s.iloc[-1])

            if any(np.isnan([last_sma20, last_rsi, last_low, last_up])):
                continue

            score = score_asset(last_close, last_sma20, last_rsi, last_low, last_up)
            sig = signal_label(score)

            rows.append(
                {
                    "Symbol": symbol,
                    "Last": last_close,
                    "Change%": chg_pct,
                    "RSI14": last_rsi,
                    "SMA20": last_sma20,
                    "BB_Lower": last_low,
                    "BB_Upper": last_up,
                    "Score": score,
                    "Signal": sig,
                }
            )

        except (ccxt.RequestTimeout, ccxt.NetworkError):
            status.warning(f"Timeout/Network issue on {symbol}")
        except ccxt.ExchangeError:
            status.info(f"Exchange error on {symbol}")
        except Exception:
            status.info(f"Skipped {symbol}")

        time.sleep(0.05)

    progress.progress(100, text="Scan complete.")
    status.empty()

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["SignalClass"] = np.select(
        [out["Score"] >= STRONG_LONG_MIN, out["Score"] <= STRONG_SHORT_MAX],
        [2, 0],
        default=1,
    )
    out = out.sort_values(["SignalClass", "Score"], ascending=[False, False]).drop(columns=["SignalClass"])
    return out.reset_index(drop=True)


def maybe_send_alerts(df: pd.DataFrame, timeframe: str, enable_telegram: bool, tg_token: str, tg_chat_id: str, cooldown_minutes: int):
    if df is None or df.empty or not enable_telegram:
        return
    if not tg_token.strip() or not tg_chat_id.strip():
        st.warning("Telegram açık ama token/chat_id boş.")
        return

    alert_state = st.session_state["alert_state"]
    candidates = df[(df["Score"] >= STRONG_LONG_MIN) | (df["Score"] <= STRONG_SHORT_MAX)].copy()
    if candidates.empty:
        return

    errors = []
    sent_any = False

    for _, row in candidates.iterrows():
        symbol = str(row["Symbol"])
        sig = str(row["Signal"])
        if not should_alert(alert_state, symbol, sig, cooldown_minutes):
            continue

        msg = build_alert_message(row, timeframe)
        ok, info = telegram_send_message(tg_token.strip(), tg_chat_id.strip(), msg)
        if not ok:
            errors.append(f"{symbol}: {info}")

        mark_alert(alert_state, symbol, sig)
        sent_any = True

    if sent_any:
        st.toast("✅ Telegram bildirim kontrolü tamamlandı (cooldown aktif).", icon="🔔")
    if errors:
        with st.expander("Telegram errors"):
            for e in errors[:30]:
                st.write(f"- {e}")


# -----------------------------
# Header
# -----------------------------
left, right = st.columns([2, 1], vertical_alignment="center")
with left:
    st.title("KuCoin Quant Sniper (Spot Market) — Top 100 USDT Pairs")
    st.caption("RSI(14), Bollinger(20,2), SMA(20) • Sniper Score (0–100) • CCXT (Rate Limit On)")
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
    st.subheader("Controls")
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "4h", "1d"], index=1)
    limit = st.slider("Candles (limit)", min_value=100, max_value=500, value=CANDLE_LIMIT, step=50)
    scan_btn = st.button("🚀 Scan Top 100", use_container_width=True)

    st.write("---")
    st.subheader("Auto Scan")
    auto_scan = st.toggle("Enable Auto Scan", value=False)
    refresh_sec = st.slider("Auto Scan Interval (sec)", 15, 600, 60, step=15)

    st.write("---")
    st.subheader("Telegram (Kolay Mod)")
    enable_telegram = st.toggle("Enable Telegram", value=False)

    # Token: password type (won't show plain)
    tg_token_in = st.text_input("Telegram Bot Token", value=st.session_state["tg_token"], type="password")
    tg_chat_in = st.text_input("Chat ID", value=st.session_state["tg_chat_id"])
    cooldown_minutes = st.slider("Cooldown (min)", 1, 180, 30, step=1)

    # Persist in session (only in this browser session)
    st.session_state["tg_token"] = tg_token_in
    st.session_state["tg_chat_id"] = tg_chat_in

    test_btn = st.button("🔔 Test Notification", use_container_width=True)

# Auto refresh
if auto_scan:
    try_autorefresh(interval_ms=int(refresh_sec * 1000), key="kucoin_auto_refresh")

# Test notification
if test_btn:
    if not st.session_state["tg_token"].strip():
        st.error("Token boş. Sidebar’a token yapıştır.")
    elif not st.session_state["tg_chat_id"].strip():
        st.error("Chat ID boş. 1358384022 yaz.")
    else:
        msg = f"✅ Test Notification\nTime(IST): {datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M:%S')}\nTF: {timeframe}"
        ok, info = telegram_send_message(st.session_state["tg_token"].strip(), st.session_state["tg_chat_id"].strip(), msg)
        if ok:
            st.success("Test mesajı Telegram’a gönderildi.")
        else:
            st.error(f"Test failed: {info}")

# Decide scanning
do_scan = False
if scan_btn:
    do_scan = True
elif auto_scan:
    last_scan = st.session_state.get("last_scan_time")
    if last_scan is None:
        do_scan = True
    else:
        if datetime.now(IST_TZ) - last_scan >= timedelta(seconds=refresh_sec):
            do_scan = True

# Run scan
if do_scan:
    try:
        syms = load_usdt_spot_symbols()
        if not syms:
            st.error("No USDT spot symbols found.")
        else:
            df_out = run_scan(syms, timeframe, limit)
            st.session_state["results_df"] = df_out
            st.session_state["last_scan_time"] = datetime.now(IST_TZ)
            st.session_state["last_scan_tf"] = timeframe

            maybe_send_alerts(
                df=df_out,
                timeframe=timeframe,
                enable_telegram=enable_telegram,
                tg_token=st.session_state["tg_token"],
                tg_chat_id=st.session_state["tg_chat_id"],
                cooldown_minutes=cooldown_minutes,
            )
    except Exception as e:
        st.error(f"Scan failed: {type(e).__name__}: {e}")

# Metrics
df_res = st.session_state.get("results_df")

info_row = st.columns([1, 2, 1, 1])
with info_row[0]:
    if st.session_state["last_scan_time"]:
        st.metric("Last Scan (IST)", st.session_state["last_scan_time"].strftime("%H:%M:%S"))
    else:
        st.metric("Last Scan (IST)", "—")
with info_row[1]:
    st.metric("Timeframe", st.session_state["last_scan_tf"] or timeframe)
with info_row[2]:
    st.metric("Assets Scored", str(len(df_res)) if isinstance(df_res, pd.DataFrame) and not df_res.empty else "0")
with info_row[3]:
    if isinstance(df_res, pd.DataFrame) and not df_res.empty:
        longs = int((df_res["Score"] >= STRONG_LONG_MIN).sum())
        shorts = int((df_res["Score"] <= STRONG_SHORT_MAX).sum())
        st.metric("🔥 Long / 💀 Short", f"{longs} / {shorts}")
    else:
        st.metric("🔥 Long / 💀 Short", "—")

st.write("")

# Tabs
if isinstance(df_res, pd.DataFrame) and not df_res.empty:
    cols = ["Symbol", "Last", "Change%", "RSI14", "SMA20", "BB_Lower", "BB_Upper", "Score", "Signal"]
    df_show = df_res.loc[:, cols].copy()

    tab_long, tab_short, tab_watch, tab_all = st.tabs(["🔥 Longs", "💀 Shorts", "⏳ Watching", "📋 All"])

    with tab_long:
        d = df_show[df_show["Score"] >= STRONG_LONG_MIN].copy()
        if d.empty:
            st.info("No STRONG LONG signals right now.")
        else:
            st.dataframe(style_scores(d), use_container_width=True, height=650)

    with tab_short:
        d = df_show[df_show["Score"] <= STRONG_SHORT_MAX].copy()
        if d.empty:
            st.info("No STRONG SHORT signals right now.")
        else:
            st.dataframe(style_scores(d), use_container_width=True, height=650)

    with tab_watch:
        d = df_show[(df_show["Score"] < STRONG_LONG_MIN) & (df_show["Score"] > STRONG_SHORT_MAX)].copy()
        st.dataframe(style_scores(d), use_container_width=True, height=650)

    with tab_all:
        st.dataframe(style_scores(df_show), use_container_width=True, height=650)
else:
    st.info("Click **🚀 Scan Top 100** (or enable **Auto Scan**) to generate signals.")
