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


# =========================
# FINAL BASE (KuCoin + OKX)
# =========================
IST_TZ = ZoneInfo("Europe/Istanbul")

TF = "15m"
HTF = "1h"
AUTO_REFRESH_SEC = 240

TOP_N_PER_EXCHANGE = 150          # scan universe per exchange
TABLE_ROWS = 20                   # final table size
FALLBACK_LONG = 10                # when no STRONG, show 10 long + 10 short
FALLBACK_SHORT = 10

STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10

SCORE_STEP = 5                    # quantize score by 5 (0,5,10,...)

RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIOD = 20
ADX_PERIOD = 14

ADX_MIN = 18.0
HTF_RSI_LONG_MAX = 60.0
HTF_RSI_SHORT_MIN = 40.0

NEAR_LONG = 70                    # only then fetch HTF to save calls
NEAR_SHORT = 30

MIN_QV_USDT = 10_000.0            # soft liquidity floor
CCXT_TIMEOUT_MS = 20000


# =========================
# PURE MATH INDICATORS
# =========================
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
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=high.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=high.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (plus_dm_smooth / tr_smooth.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm_smooth / tr_smooth.replace(0.0, np.nan))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)


def round_to_step(x: float, step: int) -> int:
    if step <= 1:
        return int(round(x))
    return int(round(x / step) * step)


# =========================
# SCORING (BASELINE)
# =========================
def raw_score(close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float) -> int:
    # Start 50, Trend +/-20, RSI +/-40, BB +/-40
    score = 50

    # Trend (20)
    if close > sma20_v:
        score += 20
    else:
        score -= 20

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


def direction_from_score(score: int) -> str:
    return "LONG" if score >= 50 else "SHORT"


def compute_gates(
    direction: str,
    close: float,
    sma20_v: float,
    rsi_v: float,
    bb_low: float,
    bb_up: float,
    adx_v: float,
    htf_close: float | None,
    htf_sma20: float | None,
    htf_rsi: float | None,
) -> int:
    # 6 KAPI
    if direction == "LONG":
        g1 = close > sma20_v
        g2 = rsi_v <= 35.0
        g3 = close <= bb_low
        g4 = adx_v >= ADX_MIN
        g5 = (htf_close is not None and htf_sma20 is not None and htf_close >= htf_sma20)
        g6 = (htf_rsi is not None and htf_rsi <= HTF_RSI_LONG_MAX)
    else:
        g1 = close < sma20_v
        g2 = rsi_v >= 65.0
        g3 = close >= bb_up
        g4 = adx_v >= ADX_MIN
        g5 = (htf_close is not None and htf_sma20 is not None and htf_close <= htf_sma20)
        g6 = (htf_rsi is not None and htf_rsi >= HTF_RSI_SHORT_MIN)

    return int(sum([g1, g2, g3, g4, g5, g6]))


def is_strong(direction: str, score: int, kapi: int) -> bool:
    if kapi != 6:
        return False
    if direction == "LONG":
        return score >= STRONG_LONG_MIN
    return score <= STRONG_SHORT_MAX


# =========================
# EXCHANGES
# =========================
def make_exchange(name: str) -> ccxt.Exchange:
    if name == "kucoin":
        return ccxt.kucoin({"enableRateLimit": True, "timeout": CCXT_TIMEOUT_MS})
    if name == "okx":
        return ccxt.okx({"enableRateLimit": True, "timeout": CCXT_TIMEOUT_MS})
    raise ValueError("unknown exchange")


def safe_fetch_tickers(ex: ccxt.Exchange) -> dict:
    try:
        return ex.fetch_tickers()
    except Exception:
        return {}


def get_top_usdt_spot_pairs(ex: ccxt.Exchange, top_n: int) -> tuple[list[str], dict]:
    markets = ex.load_markets()
    tickers = safe_fetch_tickers(ex)

    rows: list[tuple[str, float]] = []
    for sym, m in markets.items():
        try:
            if not m or not m.get("active", True):
                continue
            if not m.get("spot", False):
                continue
            if m.get("quote") != "USDT":
                continue

            t = tickers.get(sym) or {}
            qv = t.get("quoteVolume", None)

            if qv is None:
                bv = t.get("baseVolume", None)
                last = t.get("last", None)
                if bv is not None and last is not None:
                    qv = float(bv) * float(last)

            qv = float(qv) if qv is not None else 0.0
            rows.append((sym, qv))
        except Exception:
            continue

    rows.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, _ in rows[:top_n]]
    qv_map = {s: qv for s, qv in rows}
    return top, qv_map


def fetch_ohlcv_df(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 220) -> pd.DataFrame | None:
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < max(SMA_PERIOD, BB_PERIOD, RSI_PERIOD, ADX_PERIOD) + 10:
            return None
        return pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    except Exception:
        return None


def base_coin(symbol: str) -> str:
    # "BTC/USDT" -> "BTC"
    return symbol.split("/")[0].strip().upper()


# =========================
# RANKING / TABLE BUILD
# =========================
def pick_best_row_per_coin(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Build a deterministic "quality" sort key
    # strong first, then BOTH, then higher KAPI, then score direction priority, then higher QV
    src_rank = {"BOTH": 0, "KUCOIN": 1, "OKX": 2}

    tmp = df.copy()
    tmp["__src_rank"] = tmp["SOURCE"].map(src_rank).fillna(9).astype(int)
    tmp["__strong_rank"] = tmp["STRONG"].astype(bool).astype(int)

    # For LONG: higher score better; SHORT: lower score better
    tmp["__score_rank"] = np.where(
        tmp["YÖN"].eq("SHORT"),
        tmp["SKOR"].astype(int),
        -tmp["SKOR"].astype(int),
    )

    tmp = tmp.sort_values(
        by=["__strong_rank", "__src_rank", "KAPI", "__score_rank", "QV_24H"],
        ascending=[False, True, False, True, False],
        kind="mergesort",
    )

    # keep first occurrence per COIN
    tmp = tmp.drop_duplicates(subset=["COIN"], keep="first")

    return tmp.drop(columns=["__src_rank", "__strong_rank", "__score_rank"])


def build_final_table(df_all: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Table logic:
      - Prefer STRONG rows first (KAPI=6 & thresholds)
      - If not enough rows, fill remaining with best TOP candidates
      - Always try to show 10 LONG + 10 SHORT when NO STRONG exists
    """
    if df_all.empty:
        return df_all, "Aday yok (network/KuCoin/OKX veya filtre çok sert). Bir sonraki auto refresh’i bekle."

    df = df_all.copy()

    strong_df = df[df["STRONG"].astype(bool)].copy()

    if not strong_df.empty:
        # Strong exists: take all strong first, then fill with best remaining candidates
        df_sorted = df.copy()
        df_sorted = pick_best_row_per_coin(df_sorted)

        # Put STRONG + BOTH first (only ordering, no logic change)
        src_rank = {"BOTH": 0, "KUCOIN": 1, "OKX": 2}
        df_sorted["__src_rank"] = df_sorted["SOURCE"].map(src_rank).fillna(9).astype(int)
        df_sorted["__strong_rank"] = df_sorted["STRONG"].astype(bool).astype(int)
        df_sorted["__score_rank"] = np.where(df_sorted["YÖN"].eq("SHORT"), df_sorted["SKOR"], 100 - df_sorted["SKOR"])

        df_sorted = df_sorted.sort_values(
            by=["__strong_rank", "__src_rank", "KAPI", "__score_rank", "QV_24H"],
            ascending=[False, True, False, True, False],
            kind="mergesort",
        ).drop(columns=["__src_rank", "__strong_rank", "__score_rank"])

        out = df_sorted.head(TABLE_ROWS).reset_index(drop=True)
        msg = "✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu."
        return out, msg

    # No strong: enforce 10 LONG + 10 SHORT (closest)
    df_best = pick_best_row_per_coin(df)

    longs = df_best[df_best["YÖN"].eq("LONG")].sort_values(["SKOR", "KAPI", "QV_24H"], ascending=[False, False, False]).head(FALLBACK_LONG)
    shorts = df_best[df_best["YÖN"].eq("SHORT")].sort_values(["SKOR", "KAPI", "QV_24H"], ascending=[True, False, False]).head(FALLBACK_SHORT)

    out = pd.concat([longs, shorts], ignore_index=True)

    # final ordering: BOTH first (for safety), then KAPI, then score priority, then QV
    src_rank = {"BOTH": 0, "KUCOIN": 1, "OKX": 2}
    out["__src_rank"] = out["SOURCE"].map(src_rank).fillna(9).astype(int)
    out["__score_rank"] = np.where(out["YÖN"].eq("SHORT"), out["SKOR"].astype(int), -out["SKOR"].astype(int))

    out = out.sort_values(
        by=["__src_rank", "KAPI", "__score_rank", "QV_24H"],
        ascending=[True, False, True, False],
        kind="mergesort",
    ).drop(columns=["__src_rank", "__score_rank"])

    out = out.head(TABLE_ROWS).reset_index(drop=True)
    msg = "⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu."
    return out, msg


# =========================
# STYLING (NO BROKEN CSS)
# =========================
def style_table(df: pd.DataFrame):
    """
    LONG: green, SHORT: red
    STRONG rows darker
    BOTH rows slightly darker (priority)
    """
    def row_styles(row: pd.Series):
        y = str(row.get("YÖN", "")).upper()
        strong = bool(row.get("STRONG", False))
        src = str(row.get("SOURCE", "")).upper()

        # base colors
        if y == "LONG":
            base = "#0d3b2e"     # dark green
            strong_c = "#064a32" # darker green
        else:
            base = "#4b1111"     # dark red
            strong_c = "#3a0b0b" # darker red

        # BOTH emphasis (slightly darker than base, but STRONG still wins)
        both_overlay = "#0a2f26" if y == "LONG" else "#3a0f0f"

        bg = strong_c if strong else (both_overlay if src == "BOTH" else base)

        # return same style for all columns
        return [f"background-color: {bg}; color: #e6edf3;"] * len(row)

    fmt = {
        "SKOR": "{:.0f}",
        "FİYAT": "{:.6f}",
        "RAW": "{:.0f}",
        "QV_24H": "{:,.0f}",
        "KAPI": "{:.0f}",
    }

    # safer: keep simple CSS, avoid invalid tuples
    sty = (
        df.style
        .format(fmt)
        .apply(row_styles, axis=1)
        .set_properties(**{"border-color": "#1f2a37"})
    )
    return sty


# =========================
# STREAMLIT UI (NO SIDEBAR)
# =========================
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

# Auto refresh (no manual inputs)
try:
    st.autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
except Exception:
    try:
        st.experimental_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
    except Exception:
        pass

now_ist = datetime.now(IST_TZ)

st.title("KuCoin PRO Sniper — Auto (LONG + SHORT)")
st.caption(
    f"TF={TF} • HTF={HTF} • STRONG: SKOR≥{STRONG_LONG_MIN} (LONG) / SKOR≤{STRONG_SHORT_MAX} (SHORT) • "
    f"6 Kapı • Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s"
)
st.write(f"**İstanbul Time:** {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")

# Placeholders (so screen is not empty)
status_cols = st.columns(2)
kucoin_box = status_cols[0].empty()
okx_box = status_cols[1].empty()

progress_bar = st.progress(0, text="Hazırlanıyor…")
scan_box = st.empty()
msg_box = st.empty()
counts_box = st.empty()

table_header = st.markdown("## 🎯 SNIPER TABLO")

# =========================
# SCAN LOOP
# =========================
def scan_exchange(ex_name: str):
    ex = make_exchange(ex_name)
    pairs, qv_map = get_top_usdt_spot_pairs(ex, TOP_N_PER_EXCHANGE)

    rows = []
    total = len(pairs)

    for i, sym in enumerate(pairs, start=1):
        # UI heartbeat
        if i % 3 == 0 or i == 1:
            progress_bar.progress(int((i - 1) / max(total, 1) * 100), text=f"Taranıyor: {i}/{total} • {sym}")
            scan_box.info(f"Taranıyor: {i}/{total} • {sym}")

        # liquidity floor
        qv = float(qv_map.get(sym, 0.0) or 0.0)
        if qv < MIN_QV_USDT:
            continue

        df_tf = fetch_ohlcv_df(ex, sym, TF, limit=220)
        if df_tf is None:
            continue

        close = df_tf["close"].astype(float)
        high = df_tf["high"].astype(float)
        low = df_tf["low"].astype(float)

        sma20_s = sma(close, SMA_PERIOD)
        _, bb_up_s, bb_low_s = bollinger(close, BB_PERIOD, BB_STD)
        rsi_s = rsi_wilder(close, RSI_PERIOD)
        adx_s = adx_wilder(high, low, close, ADX_PERIOD)

        last_close = float(close.iloc[-1])
        last_sma20 = float(sma20_s.iloc[-1])
        last_rsi = float(rsi_s.iloc[-1])
        last_bb_low = float(bb_low_s.iloc[-1])
        last_bb_up = float(bb_up_s.iloc[-1])
        last_adx = float(adx_s.iloc[-1])

        if any(np.isnan([last_sma20, last_rsi, last_bb_low, last_bb_up, last_adx])):
            continue

        raw = raw_score(last_close, last_sma20, last_rsi, last_bb_low, last_bb_up)
        score = round_to_step(raw, SCORE_STEP)
        direction = direction_from_score(score)

        # HTF fetch only for "near" candidates
        htf_close = htf_sma20 = htf_rsi = None
        if score >= NEAR_LONG or score <= NEAR_SHORT:
            df_htf = fetch_ohlcv_df(ex, sym, HTF, limit=220)
            if df_htf is not None:
                h_close = df_htf["close"].astype(float)
                h_sma20 = sma(h_close, SMA_PERIOD)
                h_rsi = rsi_wilder(h_close, RSI_PERIOD)

                htf_close = float(h_close.iloc[-1])
                htf_sma20 = float(h_sma20.iloc[-1])
                htf_rsi = float(h_rsi.iloc[-1]) if not np.isnan(float(h_rsi.iloc[-1])) else None

        kapi = compute_gates(
            direction=direction,
            close=last_close,
            sma20_v=last_sma20,
            rsi_v=last_rsi,
            bb_low=last_bb_low,
            bb_up=last_bb_up,
            adx_v=last_adx,
            htf_close=htf_close,
            htf_sma20=htf_sma20,
            htf_rsi=htf_rsi,
        )

        strong = is_strong(direction, score, kapi)

        rows.append(
            {
                "COIN": base_coin(sym),
                "YÖN": direction,
                "SKOR": int(score),
                "FİYAT": float(last_close),
                "RAW": int(raw),
                "QV_24H": float(qv),
                "KAPI": int(kapi),
                "STRONG": bool(strong),
                "EXCHANGE": ex_name.upper(),
            }
        )

        # light sleep to be nice to rate-limit
        time.sleep(0.01)

    return rows, total


def run_dual_scan():
    # KuCoin
    try:
        kucoin_box.success("KuCoin: ✅ Bağlandı")
        ku_rows, ku_total = scan_exchange("kucoin")
    except Exception:
        kucoin_box.error("KuCoin: ❌ Hata")
        ku_rows, ku_total = [], 0

    # OKX
    try:
        okx_box.success("OKX: ✅ Bağlandı")
        ok_rows, ok_total = scan_exchange("okx")
    except Exception:
        okx_box.error("OKX: ❌ Hata")
        ok_rows, ok_total = [], 0

    return ku_rows, ok_rows, ku_total, ok_total


with st.spinner("⏳ KuCoin + OKX taranıyor…"):
    ku_rows, ok_rows, ku_total, ok_total = run_dual_scan()

progress_bar.progress(100, text="Tarama bitti ✅")
scan_box.success("Tarama bitti ✅")

df_k = pd.DataFrame(ku_rows)
df_o = pd.DataFrame(ok_rows)

# Universe size info
universe_k = len(df_k) if not df_k.empty else 0
universe_o = len(df_o) if not df_o.empty else 0
msg_box.info(f"🧠 Evren (USDT spot): KuCoin {universe_k} • OKX {universe_o} (Likidite filtresi sonrası)")

if df_k.empty and df_o.empty:
    st.warning("Aday yok (network/KuCoin/OKX veya filtre çok sert). Bir sonraki auto refresh’i bekle.")
    st.stop()

# mark BOTH (coin exists in both candidate sets)
coins_k = set(df_k["COIN"].tolist()) if not df_k.empty else set()
coins_o = set(df_o["COIN"].tolist()) if not df_o.empty else set()
both_coins = coins_k.intersection(coins_o)

def add_source(df: pd.DataFrame):
    if df.empty:
        return df
    d = df.copy()
    d["SOURCE"] = np.where(d["COIN"].isin(both_coins), "BOTH", d["EXCHANGE"])
    d = d.drop(columns=["EXCHANGE"])
    return d

df_all = pd.concat([add_source(df_k), add_source(df_o)], ignore_index=True)
df_all = pick_best_row_per_coin(df_all)

# build final table (strong first, fallback 10/10)
df_show, banner_msg = build_final_table(df_all)

# Counts (based on df_show table)
strong_long = int(((df_show["STRONG"] == True) & (df_show["YÖN"] == "LONG")).sum()) if not df_show.empty else 0
strong_short = int(((df_show["STRONG"] == True) & (df_show["YÖN"] == "SHORT")).sum()) if not df_show.empty else 0
longs = int((df_show["YÖN"] == "LONG").sum()) if not df_show.empty else 0
shorts = int((df_show["YÖN"] == "SHORT").sum()) if not df_show.empty else 0

# Banner + counts
if "STRONG bulundu" in banner_msg:
    st.success(banner_msg)
else:
    st.warning(banner_msg)

counts_box.info(
    f"✅ STRONG LONG: {strong_long} | 💀 STRONG SHORT: {strong_short} | LONG: {longs} | SHORT: {shorts}"
)

# Final reorder request: STRONG + BOTH first, then rest (ONLY ordering)
if df_show is not None and not df_show.empty:
    src_rank = {"BOTH": 0, "KUCOIN": 1, "OKX": 2}
    df_show["__src_rank"] = df_show["SOURCE"].map(src_rank).fillna(9).astype(int)
    df_show["__strong_rank"] = df_show["STRONG"].astype(bool).astype(int)
    df_show["__score_rank"] = np.where(
        df_show["YÖN"].eq("SHORT"),
        df_show["SKOR"].astype(int),
        -df_show["SKOR"].astype(int),
    )
    df_show = (
        df_show.sort_values(
            by=["__strong_rank", "__src_rank", "KAPI", "__score_rank", "QV_24H"],
            ascending=[False, True, False, True, False],
            kind="mergesort",
        )
        .drop(columns=["__src_rank", "__strong_rank", "__score_rank"])
        .reset_index(drop=True)
    )

# Display table
cols = ["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI", "STRONG", "SOURCE"]
df_show = df_show.loc[:, cols].copy()

try:
    st.dataframe(style_table(df_show), use_container_width=True, height=680)
except Exception:
    # fallback without styling if Streamlit/Pandas complains
    st.dataframe(df_show, use_container_width=True, height=680)
