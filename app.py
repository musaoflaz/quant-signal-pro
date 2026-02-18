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


# ============================================================
# FINAL BASELINE + "LEVEL 2" SAFETY GATES (NO UI/LOGIC CHANGES)
# - Keeps: dark theme, 5-step score, LONG/SHORT colors, STRONG darker,
#          top-fill, counters, auto-run (no click needed)
# - Adds: Liquidity/Spread filter, ADX trend-strength filter, ATR spike filter,
#         Higher-TF (1h) trend confirmation
# ============================================================

# -----------------------------
# Config
# -----------------------------
IST_TZ = ZoneInfo("Europe/Istanbul")

TIMEFRAME = "15m"         # baseline TF
HTF_TIMEFRAME = "1h"      # higher timeframe confirmation
CANDLE_LIMIT = 200
HTF_CANDLE_LIMIT = 200

UNIVERSE_N = 200          # practical (rate-limit friendly). Increase if you insist.
TOP_TABLE_N = 20

SCORE_STEP = 5            # 5'lik skor
STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10

# "Regular" classes (for counters / coloring)
LONG_MIN = 70
SHORT_MAX = 30

# Safety gates (Level 2)
MIN_QUOTE_VOL_USDT = 200_000     # 24h quote volume minimum
MAX_SPREAD_BPS = 40             # max spread (bps)
ADX_PERIOD = 14
ADX_STRONG = 25                 # above this is "strong trend"
ATR_PERIOD = 14
MAX_ATR_PCT = 4.5               # ATR% > this => filter/downgrade

# CCXT
EX_TIMEOUT_MS = 20000


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


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # True Range (Wilder)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan))

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


def round_to_step(x: float, step: int) -> int:
    return int(max(0, min(100, round(x / step) * step)))


# -----------------------------
# Core scoring (baseline)
# -----------------------------
def baseline_raw_score(close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float) -> float:
    # Base 50, then +/- as baseline logic
    score = 50.0

    # Trend Filter (20)
    score += 20.0 if close > sma20_v else -20.0

    # Momentum Filter (40)
    if rsi_v < 35.0:
        score += 40.0
    elif rsi_v > 65.0:
        score -= 40.0

    # Volatility Filter (40)
    if close <= bb_low:
        score += 40.0
    elif close >= bb_up:
        score -= 40.0

    return float(max(0.0, min(100.0, score)))


def label_from_score(score: int) -> str:
    if score >= STRONG_LONG_MIN:
        return "🔥 STRONG LONG"
    if score <= STRONG_SHORT_MAX:
        return "💀 STRONG SHORT"
    if score >= LONG_MIN:
        return "🟢 LONG"
    if score <= SHORT_MAX:
        return "🔴 SHORT"
    return "⏳ WATCH"


# -----------------------------
# Safety gates (Level 2)
# -----------------------------
def calc_spread_bps(bid: float | None, ask: float | None, last: float | None) -> float:
    try:
        if bid is None or ask is None:
            return 9999.0
        bid = float(bid)
        ask = float(ask)
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return 9999.0
        return ((ask - bid) / mid) * 10000.0
    except Exception:
        return 9999.0


def apply_level2_filters(
    raw: float,
    close: float,
    adx_v: float,
    plus_di: float,
    minus_di: float,
    atr_pct: float,
    htf_close: float,
    htf_sma20: float,
    quote_vol: float,
    spread_bps: float,
) -> tuple[float, str]:
    """
    Returns (adjusted_raw_score, filter_note)
    filter_note used for debugging/explainability in table.
    """
    note = []

    # Gate 1: Liquidity
    if quote_vol < MIN_QUOTE_VOL_USDT:
        # Too illiquid -> neutralize toward 50 (prevents fake STRONG on junk)
        raw = 50.0 + (raw - 50.0) * 0.25
        note.append("LOW_LIQ")

    # Gate 2: Spread
    if spread_bps > MAX_SPREAD_BPS:
        raw = 50.0 + (raw - 50.0) * 0.25
        note.append("WIDE_SPR")

    # Determine intended direction from raw
    intended_long = raw >= 60.0
    intended_short = raw <= 40.0

    # Gate 3: ATR spike filter (news/spike candles)
    if atr_pct > MAX_ATR_PCT:
        raw = 50.0 + (raw - 50.0) * 0.35
        note.append("SPIKY_ATR")

    # Gate 4: ADX trend-strength "anti-countertrend"
    # If strong trend and we try to fade it, downgrade
    if adx_v >= ADX_STRONG:
        if intended_long and (minus_di > plus_di):  # strong downtrend
            raw = 50.0 + (raw - 50.0) * 0.35
            note.append("STR_TREND_DOWN")
        if intended_short and (plus_di > minus_di):  # strong uptrend
            raw = 50.0 + (raw - 50.0) * 0.35
            note.append("STR_TREND_UP")

    # Gate 5: Higher timeframe confirmation (1h trend)
    # Long only if HTF above SMA20, Short only if HTF below SMA20 (soft gate)
    if not np.isnan(htf_sma20) and not np.isnan(htf_close):
        if intended_long and (htf_close < htf_sma20):
            raw = 50.0 + (raw - 50.0) * 0.50
            note.append("HTF_AGAINST")
        if intended_short and (htf_close > htf_sma20):
            raw = 50.0 + (raw - 50.0) * 0.50
            note.append("HTF_AGAINST")

    raw = float(max(0.0, min(100.0, raw)))
    return raw, ("|".join(note) if note else "")


# -----------------------------
# KuCoin / CCXT
# -----------------------------
def make_exchange() -> ccxt.kucoin:
    return ccxt.kucoin({"enableRateLimit": True, "timeout": EX_TIMEOUT_MS})


@st.cache_data(show_spinner=False, ttl=3600)
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


def safe_fetch_tickers(ex: ccxt.Exchange) -> dict:
    try:
        return ex.fetch_tickers()
    except Exception:
        return {}


def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> list:
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def liquidity_rank_from_ticker(t: dict) -> float:
    if not isinstance(t, dict):
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


# -----------------------------
# Selection logic for top table (STRONG first, then best candidates)
# -----------------------------
def pick_top_table(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    # STRONG sets
    strong_long = df[df["Score"] >= STRONG_LONG_MIN].sort_values("Score", ascending=False)
    strong_short = df[df["Score"] <= STRONG_SHORT_MAX].sort_values("Score", ascending=True)

    picked = pd.concat([strong_long, strong_short], axis=0).drop_duplicates(subset=["Symbol"])
    if len(picked) >= n:
        return picked.head(n).reset_index(drop=True)

    # Candidates closest to gates
    rest = df[~df["Symbol"].isin(picked["Symbol"])].copy()

    # Long side: closest to 90 (but below strong)
    long_cand = rest[rest["Score"] > 50].copy()
    long_cand["Dist"] = (STRONG_LONG_MIN - long_cand["Score"]).abs()
    long_cand = long_cand.sort_values(["Dist", "Score"], ascending=[True, False])

    # Short side: closest to 10 (but above strong)
    short_cand = rest[rest["Score"] < 50].copy()
    short_cand["Dist"] = (short_cand["Score"] - STRONG_SHORT_MAX).abs()
    short_cand = short_cand.sort_values(["Dist", "Score"], ascending=[True, True])

    # Fill remaining with best distances, keeping both sides present if possible
    remaining = n - len(picked)
    out_rows = [picked]

    # Simple balance: take alternating from long/short candidates
    li = 0
    si = 0
    longs = long_cand.to_dict("records")
    shorts = short_cand.to_dict("records")

    while remaining > 0 and (li < len(longs) or si < len(shorts)):
        take_long = (li < len(longs)) and (si >= len(shorts) or (remaining % 2 == 0))
        if take_long and li < len(longs):
            out_rows.append(pd.DataFrame([longs[li]]))
            li += 1
            remaining -= 1
            continue
        if si < len(shorts):
            out_rows.append(pd.DataFrame([shorts[si]]))
            si += 1
            remaining -= 1

    out = pd.concat(out_rows, axis=0, ignore_index=True).drop(columns=["Dist"], errors="ignore")
    out = out.drop_duplicates(subset=["Symbol"]).head(n)
    return out.reset_index(drop=True)


# -----------------------------
# Styling (dark + LONG/SHORT colors + STRONG darker)
# -----------------------------
def style_table(df: pd.DataFrame):
    def cell_style(row):
        score = float(row.get("Score", 50))
        label = str(row.get("Signal", ""))

        # Base text color
        txt = "color: #e6edf3;"

        # STRONG darker fills
        if score >= STRONG_LONG_MIN:
            return "background-color: #004d00;" + txt  # darker green
        if score <= STRONG_SHORT_MAX:
            return "background-color: #4d0000;" + txt  # darker red

        # Regular LONG/SHORT fills (lighter)
        if label == "🟢 LONG":
            return "background-color: #0a3d0a;" + txt
        if label == "🔴 SHORT":
            return "background-color: #3d0a0a;" + txt

        # Watch neutral
        return "background-color: #0b0f14;" + txt

    fmt = {
        "Last": "{:.4f}",
        "Change%": "{:.2f}",
        "RSI14": "{:.2f}",
        "SMA20": "{:.4f}",
        "BB_Lower": "{:.4f}",
        "BB_Upper": "{:.4f}",
        "ADX14": "{:.2f}",
        "ATR%": "{:.2f}",
        "Spread(bps)": "{:.1f}",
        "QV_24H": "{:,.0f}",
        "RAW": "{:.1f}",
        "Score": "{:d}",
    }

    sty = (
        df.style.format(fmt)
        .apply(lambda r: [cell_style(r)] * len(r), axis=1)
        .set_properties(**{"border-color": "#1f2a37"})
        .set_table_styles(
            [
                {"selector": "th", "props": [("background-color", "#0f172a"), ("color", "#e6edf3"), ("border", "1px solid #1f2a37")]},
                {"selector": "td", "props": [("border", "1px solid #1f2a37")]},
            ]
        )
    )
    return sty


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="KuCoin PRO Sniper — Auto (LONG + SHORT)", layout="wide")

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

# Session state
if "df_last" not in st.session_state:
    st.session_state["df_last"] = None
if "last_scan" not in st.session_state:
    st.session_state["last_scan"] = None

# Header
now_ist = datetime.now(IST_TZ)
st.title("KuCoin PRO Sniper — Auto (LONG + SHORT)")
st.caption(
    f"TF: {TIMEFRAME} • HTF: {HTF_TIMEFRAME} • Score step: {SCORE_STEP} • Universe: top {UNIVERSE_N} USDT pairs by 24h QV • IST: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}"
)

# Auto-run scan immediately (no click required)
with st.spinner("Scanning markets… (please wait)"):
    try:
        ex = make_exchange()
        symbols = load_usdt_spot_symbols()

        tickers = safe_fetch_tickers(ex)
        ranked = [(s, liquidity_rank_from_ticker(tickers.get(s, {}))) for s in symbols]
        ranked.sort(key=lambda x: x[1], reverse=True)
        scan_symbols = [s for s, _ in ranked[: min(UNIVERSE_N, len(ranked))]]

        rows = []
        prog = st.progress(0, text="Starting scan…")
        status = st.empty()

        total = len(scan_symbols)
        for i, sym in enumerate(scan_symbols, start=1):
            prog.progress(int((i - 1) / max(1, total) * 100), text=f"{sym} ({i}/{total})")

            try:
                t = tickers.get(sym, {}) if isinstance(tickers, dict) else {}
                last = t.get("last")
                bid = t.get("bid")
                ask = t.get("ask")
                qv = float(t.get("quoteVolume") or 0.0)

                ohlcv = safe_fetch_ohlcv(ex, sym, TIMEFRAME, CANDLE_LIMIT)
                if not ohlcv or len(ohlcv) < 60:
                    continue

                df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
                close = df["close"].astype(float)
                high = df["high"].astype(float)
                low = df["low"].astype(float)

                # Core indicators
                sma20_s = sma(close, 20)
                _, bb_up_s, bb_low_s = bollinger_bands(close, 20, 2.0)
                rsi_s = rsi_wilder(close, 14)

                # Level 2 indicators
                atr_s = atr_wilder(high, low, close, ATR_PERIOD)
                adx_s, pdi_s, mdi_s = adx_wilder(high, low, close, ADX_PERIOD)

                # HTF confirmation
                htf = safe_fetch_ohlcv(ex, sym, HTF_TIMEFRAME, HTF_CANDLE_LIMIT)
                if not htf or len(htf) < 40:
                    continue
                hdf = pd.DataFrame(htf, columns=["ts", "open", "high", "low", "close", "volume"])
                hclose = hdf["close"].astype(float)
                hsma20 = sma(hclose, 20)

                last_close = float(close.iloc[-1])
                prev_close = float(close.iloc[-2])
                chg_pct = ((last_close - prev_close) / prev_close) * 100.0 if prev_close else 0.0

                last_sma20 = float(sma20_s.iloc[-1])
                last_rsi = float(rsi_s.iloc[-1])
                last_low = float(bb_low_s.iloc[-1])
                last_up = float(bb_up_s.iloc[-1])

                last_atr = float(atr_s.iloc[-1])
                atr_pct = (last_atr / last_close) * 100.0 if last_close else np.nan

                last_adx = float(adx_s.iloc[-1])
                last_pdi = float(pdi_s.iloc[-1])
                last_mdi = float(mdi_s.iloc[-1])

                last_htf_close = float(hclose.iloc[-1])
                last_htf_sma20 = float(hsma20.iloc[-1])

                if any(np.isnan([last_sma20, last_rsi, last_low, last_up, last_atr, last_adx, last_htf_sma20, last_htf_close])):
                    continue

                # Baseline raw
                raw = baseline_raw_score(last_close, last_sma20, last_rsi, last_low, last_up)

                # Spread bps
                spread_bps = calc_spread_bps(bid, ask, last)

                # Level 2 gates
                raw2, note = apply_level2_filters(
                    raw=raw,
                    close=last_close,
                    adx_v=last_adx,
                    plus_di=last_pdi,
                    minus_di=last_mdi,
                    atr_pct=float(atr_pct),
                    htf_close=last_htf_close,
                    htf_sma20=last_htf_sma20,
                    quote_vol=qv,
                    spread_bps=spread_bps,
                )

                score = round_to_step(raw2, SCORE_STEP)
                sig = label_from_score(score)

                rows.append(
                    {
                        "Symbol": sym,
                        "Last": last_close,
                        "Change%": chg_pct,
                        "RSI14": last_rsi,
                        "SMA20": last_sma20,
                        "BB_Lower": last_low,
                        "BB_Upper": last_up,
                        "ADX14": last_adx,
                        "ATR%": float(atr_pct),
                        "Spread(bps)": float(spread_bps),
                        "QV_24H": float(qv),
                        "RAW": float(raw2),
                        "Score": int(score),
                        "Signal": sig,
                        "Note": note,
                    }
                )

            except (ccxt.RequestTimeout, ccxt.NetworkError):
                status.warning(f"Timeout/Network on {sym}")
            except ccxt.ExchangeError:
                # exchange may fail some symbols temporarily
                pass
            except Exception:
                pass

            # small sleep to play nice with rate limits
            time.sleep(0.03)

        prog.progress(100, text="Scan complete.")
        status.empty()

        df_all = pd.DataFrame(rows)
        st.session_state["df_last"] = df_all
        st.session_state["last_scan"] = datetime.now(IST_TZ)

    except Exception as e:
        st.error(f"Scan failed: {type(e).__name__}: {e}")

df_all = st.session_state.get("df_last")
last_scan = st.session_state.get("last_scan")

# Counters (requested)
if isinstance(df_all, pd.DataFrame) and not df_all.empty:
    strong_long_n = int((df_all["Score"] >= STRONG_LONG_MIN).sum())
    strong_short_n = int((df_all["Score"] <= STRONG_SHORT_MAX).sum())
    long_n = int(((df_all["Score"] >= LONG_MIN) & (df_all["Score"] < STRONG_LONG_MIN)).sum())
    short_n = int(((df_all["Score"] <= SHORT_MAX) & (df_all["Score"] > STRONG_SHORT_MAX)).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔥 STRONG LONG", strong_long_n)
    c2.metric("💀 STRONG SHORT", strong_short_n)
    c3.metric("🟢 LONG", long_n)
    c4.metric("🔴 SHORT", short_n)

    if last_scan:
        st.caption(f"Last scan (IST): {last_scan.strftime('%Y-%m-%d %H:%M:%S')}")

    # Build TOP table: STRONG first, then best candidates (fills to 20)
    df_top = pick_top_table(df_all, TOP_TABLE_N)

    # Keep a clean table like your final (hide advanced columns if you want)
    show_cols = [
        "Signal",
        "Symbol",
        "Score",
        "Last",
        "Change%",
        "RSI14",
        "SMA20",
        "BB_Lower",
        "BB_Upper",
        "ADX14",
        "ATR%",
        "Spread(bps)",
        "QV_24H",
        "Note",
    ]
    df_show = df_top.loc[:, show_cols].copy()

    st.dataframe(style_table(df_show), use_container_width=True, height=690)

else:
    st.info("Aday yok (veri çekilemedi). Yenileyip tekrar deneyebilirsin.")
