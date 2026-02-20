# requirements.txt (install these):
# streamlit
# pandas
# numpy
# ccxt

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import ccxt


# ============================================================
# DUAL ENGINE: KUCOIN SPOT + BINANCE SPOT (NO TELEGRAM)
# - Dark theme, auto-run (no clicks), progress spinner
# - Vectorized indicators (NO pandas_ta / TA-Lib / finta)
# - Level 1 score + Level 2 safety gates
# - One row per BASE asset: KuCoin score + Binance score side-by-side
# ============================================================

IST_TZ = ZoneInfo("Europe/Istanbul")

# Timeframes
TF = "15m"
HTF = "1h"
LIMIT = 200
HTF_LIMIT = 200

# Universe / output
TOP_PER_EXCHANGE = 150
UNION_CAP = 220          # union of bases may grow; cap to keep runtime sane
TOP_TABLE_N = 20

# Score
SCORE_STEP = 5
STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10
LONG_MIN = 70
SHORT_MAX = 30

# Level 2 gates
MIN_QUOTE_VOL_USDT = 200_000
MAX_SPREAD_BPS = 40
ADX_PERIOD = 14
ADX_STRONG = 25
ATR_PERIOD = 14
MAX_ATR_PCT = 4.5

# Cross-check divergence thresholds
DIVERGENCE_LONG_OTHER_MAX = 70   # if one >=90 and other <=70 -> divergence
DIVERGENCE_SHORT_OTHER_MIN = 30  # if one <=10 and other >=30 -> divergence

# CCXT
TIMEOUT_MS = 20000
MAX_WORKERS = 8          # small pool; ccxt + rate limits
SLEEP_BETWEEN = 0.02     # gentle


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
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

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
# Scoring
# -----------------------------
def baseline_raw_score(close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float) -> float:
    score = 50.0
    score += 20.0 if close > sma20_v else -20.0

    if rsi_v < 35.0:
        score += 40.0
    elif rsi_v > 65.0:
        score -= 40.0

    if close <= bb_low:
        score += 40.0
    elif close >= bb_up:
        score -= 40.0

    return float(max(0.0, min(100.0, score)))


def calc_spread_bps(bid: float | None, ask: float | None) -> float:
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


def apply_level2_gates(
    raw: float,
    quote_vol: float,
    spread_bps: float,
    atr_pct: float,
    adx_v: float,
    plus_di: float,
    minus_di: float,
    htf_close: float,
    htf_sma20: float,
) -> tuple[float, str]:
    note = []

    # Intended direction from raw (soft)
    intended_long = raw >= 60.0
    intended_short = raw <= 40.0

    # Gate: Liquidity
    if quote_vol < MIN_QUOTE_VOL_USDT:
        raw = 50.0 + (raw - 50.0) * 0.25
        note.append("LOW_LIQ")

    # Gate: Spread
    if spread_bps > MAX_SPREAD_BPS:
        raw = 50.0 + (raw - 50.0) * 0.25
        note.append("WIDE_SPR")

    # Gate: ATR spike
    if atr_pct > MAX_ATR_PCT:
        raw = 50.0 + (raw - 50.0) * 0.35
        note.append("SPIKY_ATR")

    # Gate: ADX strong trend anti-countertrend
    if adx_v >= ADX_STRONG:
        if intended_long and (minus_di > plus_di):
            raw = 50.0 + (raw - 50.0) * 0.35
            note.append("STR_TREND_DOWN")
        if intended_short and (plus_di > minus_di):
            raw = 50.0 + (raw - 50.0) * 0.35
            note.append("STR_TREND_UP")

    # Gate: HTF confirmation (soft pull)
    if not np.isnan(htf_sma20) and not np.isnan(htf_close):
        if intended_long and (htf_close < htf_sma20):
            raw = 50.0 + (raw - 50.0) * 0.50
            note.append("HTF_AGAINST")
        if intended_short and (htf_close > htf_sma20):
            raw = 50.0 + (raw - 50.0) * 0.50
            note.append("HTF_AGAINST")

    raw = float(max(0.0, min(100.0, raw)))
    return raw, ("|".join(note) if note else "")


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
# CCXT exchanges
# -----------------------------
def ex_kucoin() -> ccxt.kucoin:
    return ccxt.kucoin({"enableRateLimit": True, "timeout": TIMEOUT_MS})


def ex_binance() -> ccxt.binance:
    return ccxt.binance(
        {
            "enableRateLimit": True,
            "timeout": TIMEOUT_MS,
            "options": {"defaultType": "spot"},
        }
    )


def safe_fetch_tickers(ex: ccxt.Exchange) -> dict:
    try:
        return ex.fetch_tickers()
    except Exception:
        return {}


def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> list:
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def quote_volume_usdt(t: dict) -> float:
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


@st.cache_data(show_spinner=False, ttl=3600)
def load_spot_usdt_markets(exchange_name: str) -> dict:
    ex = ex_kucoin() if exchange_name == "kucoin" else ex_binance()
    markets = ex.load_markets()
    out = {}
    for sym, m in markets.items():
        if not m:
            continue
        if not m.get("active", True):
            continue
        if not m.get("spot", False):
            continue
        if m.get("quote") != "USDT":
            continue
        base = m.get("base")
        quote = m.get("quote")
        if not base or quote != "USDT":
            continue
        out[sym] = {"base": base, "quote": quote}
    return out


def top_symbols_by_volume(exchange_name: str, top_n: int) -> tuple[list[str], dict]:
    ex = ex_kucoin() if exchange_name == "kucoin" else ex_binance()
    markets = load_spot_usdt_markets(exchange_name)
    tickers = safe_fetch_tickers(ex)

    ranked = []
    for sym in markets.keys():
        t = tickers.get(sym, {})
        ranked.append((sym, quote_volume_usdt(t)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    top_syms = [s for s, _ in ranked[: min(top_n, len(ranked))]]

    return top_syms, tickers


# -----------------------------
# Per-symbol scan
# -----------------------------
def scan_symbol(exchange_name: str, symbol: str, tickers: dict) -> dict | None:
    ex = ex_kucoin() if exchange_name == "kucoin" else ex_binance()

    try:
        t = tickers.get(symbol, {}) if isinstance(tickers, dict) else {}
        bid = t.get("bid")
        ask = t.get("ask")
        last = t.get("last")
        qv = float(quote_volume_usdt(t))

        ohlcv = safe_fetch_ohlcv(ex, symbol, TF, LIMIT)
        if not ohlcv or len(ohlcv) < 60:
            return None

        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        sma20_s = sma(close, 20)
        _, bb_up_s, bb_low_s = bollinger_bands(close, 20, 2.0)
        rsi_s = rsi_wilder(close, 14)

        atr_s = atr_wilder(high, low, close, ATR_PERIOD)
        adx_s, pdi_s, mdi_s = adx_wilder(high, low, close, ADX_PERIOD)

        htf = safe_fetch_ohlcv(ex, symbol, HTF, HTF_LIMIT)
        if not htf or len(htf) < 40:
            return None
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

        if any(np.isnan([last_sma20, last_rsi, last_low, last_up, last_atr, last_adx, last_htf_close, last_htf_sma20])):
            return None

        raw = baseline_raw_score(last_close, last_sma20, last_rsi, last_low, last_up)
        spread_bps = calc_spread_bps(bid, ask)

        raw2, note = apply_level2_gates(
            raw=raw,
            quote_vol=qv,
            spread_bps=float(spread_bps),
            atr_pct=float(atr_pct),
            adx_v=float(last_adx),
            plus_di=float(last_pdi),
            minus_di=float(last_mdi),
            htf_close=float(last_htf_close),
            htf_sma20=float(last_htf_sma20),
        )

        score = round_to_step(raw2, SCORE_STEP)
        sig = label_from_score(int(score))

        return {
            "symbol": symbol,
            "last": float(last_close),
            "chg": float(chg_pct),
            "rsi": float(last_rsi),
            "adx": float(last_adx),
            "atr_pct": float(atr_pct),
            "spread_bps": float(spread_bps),
            "qv": float(qv),
            "raw": float(raw2),
            "score": int(score),
            "signal": sig,
            "note": note,
        }

    except (ccxt.RequestTimeout, ccxt.NetworkError, ccxt.ExchangeError):
        return None
    except Exception:
        return None


def parallel_scan(exchange_name: str, symbols: list[str], tickers: dict, progress_cb=None) -> dict[str, dict]:
    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(scan_symbol, exchange_name, s, tickers): s for s in symbols}
        done = 0
        total = len(futs)

        for fut in as_completed(futs):
            s = futs[fut]
            out = None
            try:
                out = fut.result()
            except Exception:
                out = None

            if out is not None:
                results[s] = out

            done += 1
            if progress_cb:
                progress_cb(done, total, s)

            time.sleep(SLEEP_BETWEEN)

    return results


# -----------------------------
# Merge cross-check (one row per base)
# -----------------------------
def build_unified_table(
    ku_markets: dict,
    bn_markets: dict,
    ku_data: dict[str, dict],
    bn_data: dict[str, dict],
) -> pd.DataFrame:
    # Map base -> symbol on each exchange
    base_to_ku = {}
    for sym, meta in ku_markets.items():
        base_to_ku[meta["base"]] = sym
    base_to_bn = {}
    for sym, meta in bn_markets.items():
        base_to_bn[meta["base"]] = sym

    bases = sorted(set(base_to_ku.keys()) | set(base_to_bn.keys()))
    if len(bases) > UNION_CAP:
        # prefer bases that have data on at least one exchange
        bases_with_data = []
        for b in bases:
            ks = base_to_ku.get(b)
            bs = base_to_bn.get(b)
            if (ks and ks in ku_data) or (bs and bs in bn_data):
                bases_with_data.append(b)
        bases = bases_with_data[:UNION_CAP]

    rows = []
    for base in bases:
        ku_sym = base_to_ku.get(base)
        bn_sym = base_to_bn.get(base)

        ku = ku_data.get(ku_sym, None) if ku_sym else None
        bn = bn_data.get(bn_sym, None) if bn_sym else None

        ku_score = ku["score"] if ku else np.nan
        bn_score = bn["score"] if bn else np.nan

        source = "Both" if (ku_sym and bn_sym) else ("KuCoin-only" if ku_sym else "Binance-only")

        # Final status logic
        status = "WATCH"
        if ku is None and bn is None:
            status = "NO_DATA"
        else:
            # strong long dual
            if (not np.isnan(ku_score)) and (not np.isnan(bn_score)) and (ku_score >= STRONG_LONG_MIN) and (bn_score >= STRONG_LONG_MIN):
                status = "DUAL_CONFIRMED_LONG"
            elif (not np.isnan(ku_score)) and (not np.isnan(bn_score)) and (ku_score <= STRONG_SHORT_MAX) and (bn_score <= STRONG_SHORT_MAX):
                status = "DUAL_CONFIRMED_SHORT"
            else:
                # divergence (long side)
                if (not np.isnan(ku_score)) and (ku_score >= STRONG_LONG_MIN):
                    if (not np.isnan(bn_score)) and (bn_score <= DIVERGENCE_LONG_OTHER_MAX):
                        status = "DIVERGENCE"
                    else:
                        status = "SINGLE_STRONG"
                if (not np.isnan(bn_score)) and (bn_score >= STRONG_LONG_MIN):
                    if (not np.isnan(ku_score)) and (ku_score <= DIVERGENCE_LONG_OTHER_MAX):
                        status = "DIVERGENCE"
                    else:
                        status = "SINGLE_STRONG"

                # divergence (short side)
                if (not np.isnan(ku_score)) and (ku_score <= STRONG_SHORT_MAX):
                    if (not np.isnan(bn_score)) and (bn_score >= DIVERGENCE_SHORT_OTHER_MIN):
                        status = "DIVERGENCE"
                    else:
                        status = "SINGLE_STRONG"
                if (not np.isnan(bn_score)) and (bn_score <= STRONG_SHORT_MAX):
                    if (not np.isnan(ku_score)) and (ku_score >= DIVERGENCE_SHORT_OTHER_MIN):
                        status = "DIVERGENCE"
                    else:
                        status = "SINGLE_STRONG"

        # Provide a "best side" for display (for coloring consistency)
        # choose higher confidence: dual strong first, else max(abs(score-50))
        best_sig = "⏳ WATCH"
        best_score = np.nan
        best_ex = ""
        best_note = ""

        def strength(s: float) -> float:
            if np.isnan(s):
                return -1.0
            return abs(s - 50.0)

        if (ku is not None) and (bn is not None):
            if status.startswith("DUAL_CONFIRMED"):
                # pick one to show label; both are strong anyway
                best_ex = "Both"
                best_score = max(ku_score, bn_score)
                best_sig = "🔥 DUAL LONG" if status == "DUAL_CONFIRMED_LONG" else "💀 DUAL SHORT"
                best_note = "DUAL"
            else:
                if strength(ku_score) >= strength(bn_score):
                    best_ex, best_score, best_sig, best_note = "KuCoin", ku_score, ku["signal"], ku.get("note", "")
                else:
                    best_ex, best_score, best_sig, best_note = "Binance", bn_score, bn["signal"], bn.get("note", "")
        elif ku is not None:
            best_ex, best_score, best_sig, best_note = "KuCoin", ku_score, ku["signal"], ku.get("note", "")
        elif bn is not None:
            best_ex, best_score, best_sig, best_note = "Binance", bn_score, bn["signal"], bn.get("note", "")

        rows.append(
            {
                "Signal": best_sig,
                "Base": base,
                "Source": source,
                "Status": status,
                "KuCoin_Score": ku_score,
                "Binance_Score": bn_score,
                "Best_Exchange": best_ex,
                "Best_Score": best_score,
                "KuCoin_Note": (ku.get("note", "") if ku else ""),
                "Binance_Note": (bn.get("note", "") if bn else ""),
            }
        )

    df = pd.DataFrame(rows)
    return df


def pick_top_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df

    d = df.copy()

    # Priority ranking
    pr = np.select(
        [
            d["Status"].isin(["DUAL_CONFIRMED_LONG", "DUAL_CONFIRMED_SHORT"]),
            d["Status"].eq("SINGLE_STRONG"),
            d["Status"].eq("DIVERGENCE"),
            d["Status"].eq("NO_DATA"),
        ],
        [4, 3, 2, 0],
        default=1,  # WATCH
    )
    d["PR"] = pr

    # Distance to elite thresholds to fill when no strong
    # if best score above 50 -> closeness to 90; if below 50 -> closeness to 10
    bs = d["Best_Score"].astype(float)
    dist = np.where(
        bs >= 50,
        np.abs(STRONG_LONG_MIN - bs),
        np.abs(bs - STRONG_SHORT_MAX),
    )
    d["DIST"] = dist

    d = d.sort_values(["PR", "DIST"], ascending=[False, True])
    d = d.drop(columns=["PR", "DIST"], errors="ignore")
    return d.head(n).reset_index(drop=True)


# -----------------------------
# Styling (dark)
# -----------------------------
def style_table(df: pd.DataFrame):
    def row_style(row):
        status = str(row.get("Status", ""))
        sig = str(row.get("Signal", ""))
        best = row.get("Best_Score", np.nan)
        try:
            best = float(best)
        except Exception:
            best = np.nan

        txt = "color: #e6edf3;"

        # Dual confirmed darker
        if status == "DUAL_CONFIRMED_LONG":
            return "background-color: #003a00;" + txt
        if status == "DUAL_CONFIRMED_SHORT":
            return "background-color: #3a0000;" + txt

        # Single strong (bright-ish)
        if not np.isnan(best) and best >= STRONG_LONG_MIN:
            return "background-color: #006400;" + txt
        if not np.isnan(best) and best <= STRONG_SHORT_MAX:
            return "background-color: #8B0000;" + txt

        # Regular long/short colors
        if sig == "🟢 LONG":
            return "background-color: #0a3d0a;" + txt
        if sig == "🔴 SHORT":
            return "background-color: #3d0a0a;" + txt

        # Divergence highlight (neutral amber-like dark)
        if status == "DIVERGENCE":
            return "background-color: #2a2200;" + txt

        # Watch
        return "background-color: #0b0f14;" + txt

    fmt = {
        "KuCoin_Score": "{:.0f}",
        "Binance_Score": "{:.0f}",
        "Best_Score": "{:.0f}",
    }

    sty = (
        df.style.format(fmt)
        .apply(lambda r: [row_style(r)] * len(r), axis=1)
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
st.set_page_config(page_title="DUAL Sniper — KuCoin + Binance (Spot)", layout="wide")

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

now_ist = datetime.now(IST_TZ)
st.title("DUAL Sniper — KuCoin + Binance (Spot)")
st.caption(
    f"TF: {TF} • HTF: {HTF} • step={SCORE_STEP} • Top {TOP_PER_EXCHANGE}/exchange by QV • IST: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}"
)

# Auto-run scan on load
progress = st.progress(0, text="Starting…")
status = st.empty()

def make_progress_cb(prefix: str):
    def _cb(done: int, total: int, sym: str):
        pct = int((done / max(1, total)) * 100)
        progress.progress(min(100, pct), text=f"{prefix}: {sym} ({done}/{total})")
    return _cb

with st.spinner("Scanning both exchanges…"):
    # Prepare markets
    ku_markets = load_spot_usdt_markets("kucoin")
    bn_markets = load_spot_usdt_markets("binance")

    # Top symbols by volume + tickers
    try:
        status.info("Fetching KuCoin tickers & ranking…")
        ku_syms, ku_tickers = top_symbols_by_volume("kucoin", TOP_PER_EXCHANGE)
    except Exception:
        ku_syms, ku_tickers = [], {}

    try:
        status.info("Fetching Binance tickers & ranking…")
        bn_syms, bn_tickers = top_symbols_by_volume("binance", TOP_PER_EXCHANGE)
    except Exception:
        bn_syms, bn_tickers = [], {}

    # Scan in parallel per exchange (each task uses a fresh exchange instance)
    ku_data = {}
    bn_data = {}

    if ku_syms:
        status.info("Scanning KuCoin OHLCV…")
        ku_data = parallel_scan("kucoin", ku_syms, ku_tickers, progress_cb=make_progress_cb("KuCoin"))
    else:
        status.warning("KuCoin symbols not available (tickers/markets failed).")

    if bn_syms:
        status.info("Scanning Binance OHLCV…")
        bn_data = parallel_scan("binance", bn_syms, bn_tickers, progress_cb=make_progress_cb("Binance"))
    else:
        status.warning("Binance symbols not available (tickers/markets failed).")

progress.progress(100, text="Scan complete.")
status.empty()

# Build unified
df_unified = build_unified_table(ku_markets, bn_markets, ku_data, bn_data)
df_top = pick_top_rows(df_unified, TOP_TABLE_N)

# Counters
def count_strong_long(scores: pd.Series) -> int:
    s = pd.to_numeric(scores, errors="coerce")
    return int((s >= STRONG_LONG_MIN).sum())

def count_strong_short(scores: pd.Series) -> int:
    s = pd.to_numeric(scores, errors="coerce")
    return int((s <= STRONG_SHORT_MAX).sum())

ku_str_long = count_strong_long(df_unified["KuCoin_Score"])
ku_str_short = count_strong_short(df_unified["KuCoin_Score"])
bn_str_long = count_strong_long(df_unified["Binance_Score"])
bn_str_short = count_strong_short(df_unified["Binance_Score"])

dual_long = int((df_unified["Status"] == "DUAL_CONFIRMED_LONG").sum())
dual_short = int((df_unified["Status"] == "DUAL_CONFIRMED_SHORT").sum())

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("🔥 DUAL LONG", dual_long)
c2.metric("💀 DUAL SHORT", dual_short)
c3.metric("KuCoin 🔥/💀", f"{ku_str_long} / {ku_str_short}")
c4.metric("Binance 🔥/💀", f"{bn_str_long} / {bn_str_short}")
c5.metric("Universe (rows)", str(len(df_unified)))
c6.metric("Top Table", str(len(df_top)))

st.write("")

# Display table
if df_top.empty:
    st.info("Aday yok (veri çekilemedi). Yenileyip tekrar deneyebilirsin.")
else:
    # Keep a clean final view
    show_cols = [
        "Signal",
        "Base",
        "Source",
        "Status",
        "KuCoin_Score",
        "Binance_Score",
        "Best_Exchange",
        "Best_Score",
        "KuCoin_Note",
        "Binance_Note",
    ]
    df_show = df_top.loc[:, show_cols].copy()
    st.dataframe(style_table(df_show), use_container_width=True, height=690)
