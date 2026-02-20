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
# FINAL BASE (KuCoin + OKX) — Auto (LONG + SHORT)
# - Dark theme + 5'lik skor adımı
# - 6 kapı (TF=15m + HTF=1h) -> STRONG sadece KAPI==6 iken
# - Likidite filtresi + stablecoin filtre
# - STRONG varsa önce STRONG'lar, boş kalırsa TOP adaylarla tablo dolu (10 LONG + 10 SHORT)
# - SOURCE: KUCOIN / OKX / BOTH (BOTH üstte)
# =============================


# -----------------------------
# Config
# -----------------------------
IST_TZ = ZoneInfo("Europe/Istanbul")

TF = "15m"
HTF = "1h"

REFRESH_SEC = 240

CANDLE_LIMIT = 220
SMA_PERIOD = 20
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
ADX_PERIOD = 14

TOP_PER_EXCHANGE = 200  # kalite + hız dengesi

# STRONG eşikleri (SKOR üzerinden)
STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10

# Skor adımı
SCORE_STEP = 5

# Likidite filtresi (USDT quote volume)
MIN_QV_24H_USDT = 1_000_000

# Stablecoin/peg filtre (base)
STABLE_BASES = {
    "USDT", "USDC", "DAI", "TUSD", "USDP", "FDUSD", "BUSD", "PYUSD",
    "EURC", "USDD", "USDG", "USDE", "FRAX", "LUSD", "SUSD", "USTC",
}

# Gate/filtre eşikleri (Level 2 sertlik)
ADX_MIN_TF = 18.0
ADX_MIN_HTF = 15.0


# -----------------------------
# Pure pandas/numpy indicators
# -----------------------------
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


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    # Wilder's ADX, pure pandas
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(
        0.0, np.nan
    )
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(
        0.0, np.nan
    )

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx_v = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx_v.fillna(0.0)


def round_step(x: float, step: int) -> int:
    return int(step * round(float(x) / step))


# -----------------------------
# CCXT helpers
# -----------------------------
def make_exchange(name: str) -> ccxt.Exchange:
    params = {"enableRateLimit": True, "timeout": 20000}
    if name == "kucoin":
        return ccxt.kucoin(params)
    if name == "okx":
        return ccxt.okx(params)
    raise ValueError("unsupported exchange")


def safe_load_markets(ex: ccxt.Exchange) -> dict:
    try:
        return ex.load_markets()
    except Exception:
        return {}


def safe_fetch_tickers(ex: ccxt.Exchange, symbols: list[str]) -> dict:
    try:
        return ex.fetch_tickers(symbols)
    except Exception:
        try:
            all_t = ex.fetch_tickers()
            return {s: all_t.get(s) for s in symbols if s in all_t}
        except Exception:
            return {}


def compute_qv_24h_usdt(ticker: dict) -> float:
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


def safe_fetch_ohlcv_df(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    try:
        o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not o or len(o) < max(SMA_PERIOD, BB_PERIOD, RSI_PERIOD, ADX_PERIOD) + 10:
            return None
        df = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
        return df
    except Exception:
        return None


def get_usdt_spot_symbols(markets: dict) -> list[str]:
    syms: list[str] = []
    for sym, m in markets.items():
        if not m or not isinstance(m, dict):
            continue
        if not m.get("active", True):
            continue
        if not m.get("spot", False):
            continue
        if m.get("quote") != "USDT":
            continue
        base = m.get("base") or (sym.split("/")[0] if "/" in sym else "")
        if base in STABLE_BASES:
            continue
        syms.append(sym)
    return sorted(set(syms))


# -----------------------------
# Scoring + Gates (6 kapı)
# -----------------------------
def raw_score_from_core(close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float) -> float:
    # Base 50; +/-(20,40,40)
    score = 50.0
    # Trend (20)
    score += 20.0 if close > sma20_v else -20.0
    # Momentum (40)
    if rsi_v < 35.0:
        score += 40.0
    elif rsi_v > 65.0:
        score -= 40.0
    # Volatility (40)
    if close <= bb_low:
        score += 40.0
    elif close >= bb_up:
        score -= 40.0
    return float(np.clip(score, 0.0, 100.0))


def direction_from_raw(raw: float) -> str:
    # Long bias if >50, Short bias if <50, else Long (neutral)
    if raw >= 50.0:
        return "LONG"
    return "SHORT"


def gates_6(
    direction: str,
    close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float, adx_v: float,
    h_close: float | None, h_sma20: float | None, h_rsi: float | None, h_adx: float | None
) -> int:
    g = 0

    # Gate 1: TF trend
    if direction == "LONG" and close > sma20_v:
        g += 1
    if direction == "SHORT" and close < sma20_v:
        g += 1

    # Gate 2: TF RSI confirm (not too late)
    if direction == "LONG" and rsi_v <= 55.0:
        g += 1
    if direction == "SHORT" and rsi_v >= 45.0:
        g += 1

    # Gate 3: TF Bollinger location (near edges)
    if direction == "LONG" and close <= bb_low:
        g += 1
    if direction == "SHORT" and close >= bb_up:
        g += 1

    # Gate 4: TF ADX strength
    if adx_v >= ADX_MIN_TF:
        g += 1

    # Gate 5: HTF trend
    if h_close is not None and h_sma20 is not None:
        if direction == "LONG" and h_close > h_sma20:
            g += 1
        if direction == "SHORT" and h_close < h_sma20:
            g += 1

    # Gate 6: HTF momentum + HTF ADX
    ok_htf_momo = False
    if h_rsi is not None:
        if direction == "LONG" and h_rsi <= 60.0:
            ok_htf_momo = True
        if direction == "SHORT" and h_rsi >= 40.0:
            ok_htf_momo = True

    ok_htf_adx = False
    if h_adx is not None and h_adx >= ADX_MIN_HTF:
        ok_htf_adx = True

    if ok_htf_momo and ok_htf_adx:
        g += 1

    return g


def is_strong(direction: str, skor: int, gates: int) -> bool:
    if gates != 6:
        return False
    if direction == "LONG":
        return skor >= STRONG_LONG_MIN
    return skor <= STRONG_SHORT_MAX


# -----------------------------
# UI (dark)
# -----------------------------
st.set_page_config(page_title="KuCoin PRO Sniper — Auto (LONG + SHORT)", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }
h1, h2, h3, h4, h5, h6, p, span, div { color: #e6edf3; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
</style>
""",
    unsafe_allow_html=True,
)

# Auto refresh
try:
    st.autorefresh(interval=REFRESH_SEC * 1000, key="auto_refresh")
except Exception:
    try:
        st.experimental_autorefresh(interval=REFRESH_SEC * 1000, key="auto_refresh")
    except Exception:
        pass

now_ist = datetime.now(IST_TZ)

st.title("🎯 KuCoin PRO Sniper — Auto (LONG + SHORT)")
st.caption(f"TF={TF} • HTF={HTF} • STRONG: SKOR≥{STRONG_LONG_MIN} (LONG) / SKOR≤{STRONG_SHORT_MAX} (SHORT) • 6 Kapı • Skor adımı: {SCORE_STEP} • Auto: {REFRESH_SEC}s")
st.markdown(f"**İstanbul Time:** {now_ist.strftime('%Y-%m-%d %H:%M:%S')}")

# Connection check boxes
conn_row = st.columns(2)
kucoin_ok = False
okx_ok = False

with conn_row[0]:
    try:
        ex_k = make_exchange("kucoin")
        _ = safe_load_markets(ex_k)
        kucoin_ok = True if _ else False
    except Exception:
        kucoin_ok = False
    st.success("KuCoin: ✅ Bağlandı" if kucoin_ok else "KuCoin: ❌ Hata")

with conn_row[1]:
    try:
        ex_o = make_exchange("okx")
        _ = safe_load_markets(ex_o)
        okx_ok = True if _ else False
    except Exception:
        okx_ok = False
    st.success("OKX: ✅ Bağlandı" if okx_ok else "OKX: ❌ Hata")


def style_table(df: pd.DataFrame):
    # Row-based styling: LONG green, SHORT red; STRONG darker; BOTH slightly emphasized
    cols = list(df.columns)

    def row_css(row: pd.Series):
        direction = str(row.get("YÖN", ""))
        strong = bool(row.get("STRONG", False))
        source = str(row.get("SOURCE", ""))

        if direction == "LONG":
            bg = "#0d3b2e" if strong else "#134a3a"
        else:
            bg = "#4a0f12" if strong else "#5a1518"

        # BOTH boost: darker shade
        if source == "BOTH":
            bg = "#0a2f25" if direction == "LONG" else "#3a0c0e"

        return [f"background-color: {bg}; color: #e6edf3;"] * len(cols)

    fmt = {
        "SKOR": "{:,.0f}",
        "FİYAT": "{:.6f}",
        "RAW": "{:,.0f}",
        "QV_24H": "{:,.0f}",
        "KAPI": "{:,.0f}",
    }

    return (
        df.style.format(fmt)
        .apply(row_css, axis=1)
        .set_properties(**{"border-color": "#1f2a37"})
    )


def scan_exchange(name: str) -> tuple[pd.DataFrame, dict]:
    ex = make_exchange(name)
    markets = safe_load_markets(ex)
    syms = get_usdt_spot_symbols(markets)

    tickers = safe_fetch_tickers(ex, syms)
    rows_rank = []
    for s in syms:
        t = tickers.get(s)
        qv = compute_qv_24h_usdt(t)
        if qv >= MIN_QV_24H_USDT:
            rows_rank.append((s, qv))

    rows_rank.sort(key=lambda x: x[1], reverse=True)
    top_syms = [s for s, _ in rows_rank[:TOP_PER_EXCHANGE]]

    meta = {
        "universe": len(syms),
        "after_liquidity": len(rows_rank),
        "scanned": len(top_syms),
    }

    out = []
    for sym in top_syms:
        df = safe_fetch_ohlcv_df(ex, sym, TF, CANDLE_LIMIT)
        if df is None:
            continue

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        sma20_s = sma(close, SMA_PERIOD)
        _, bb_up_s, bb_low_s = bollinger(close, BB_PERIOD, BB_STD)
        rsi_s = rsi_wilder(close, RSI_PERIOD)
        adx_s = adx(high, low, close, ADX_PERIOD)

        last_close = float(close.iloc[-1])
        last_sma20 = float(sma20_s.iloc[-1])
        last_rsi = float(rsi_s.iloc[-1])
        last_low = float(bb_low_s.iloc[-1])
        last_up = float(bb_up_s.iloc[-1])
        last_adx = float(adx_s.iloc[-1])

        if any(np.isnan([last_sma20, last_rsi, last_low, last_up, last_adx])):
            continue

        raw = raw_score_from_core(last_close, last_sma20, last_rsi, last_low, last_up)
        direction = direction_from_raw(raw)

        # HTF ALWAYS (kritik düzeltme: KAPI 6 olabilsin)
        h_close = h_sma20 = h_rsi = h_adx = None
        dfh = safe_fetch_ohlcv_df(ex, sym, HTF, CANDLE_LIMIT)
        if dfh is not None:
            hc = dfh["close"].astype(float)
            hh = dfh["high"].astype(float)
            hl = dfh["low"].astype(float)

            hsma20 = sma(hc, SMA_PERIOD)
            hrsi = rsi_wilder(hc, RSI_PERIOD)
            hadx = adx(hh, hl, hc, ADX_PERIOD)

            try:
                h_close = float(hc.iloc[-1])
                h_sma20 = float(hsma20.iloc[-1])
                h_rsi_v = float(hrsi.iloc[-1])
                h_adx_v = float(hadx.iloc[-1])
                h_rsi = None if np.isnan(h_rsi_v) else h_rsi_v
                h_adx = None if np.isnan(h_adx_v) else h_adx_v
            except Exception:
                h_close = h_sma20 = h_rsi = h_adx = None

        gates = gates_6(direction, last_close, last_sma20, last_rsi, last_low, last_up, last_adx, h_close, h_sma20, h_rsi, h_adx)
        skor = round_step(raw, SCORE_STEP)

        # ticker qv
        qv = compute_qv_24h_usdt(tickers.get(sym))

        base = sym.split("/")[0] if "/" in sym else sym
        out.append(
            {
                "BASE": base,
                "SYM": sym,
                "YÖN": direction,
                "SKOR": int(skor),
                "FİYAT": float(last_close),
                "RAW": int(round(raw)),
                "QV_24H": float(qv),
                "KAPI": int(gates),
                "STRONG": bool(is_strong(direction, int(skor), int(gates))),
                "EX": name.upper(),
            }
        )

    return pd.DataFrame(out), meta


def merge_kucoin_okx(df_k: pd.DataFrame, df_o: pd.DataFrame) -> pd.DataFrame:
    if df_k is None:
        df_k = pd.DataFrame()
    if df_o is None:
        df_o = pd.DataFrame()

    by_base = {}

    # insert all kucoin
    for _, r in df_k.iterrows():
        base = r["BASE"]
        by_base.setdefault(base, {})["KUCOIN"] = r.to_dict()

    # insert all okx
    for _, r in df_o.iterrows():
        base = r["BASE"]
        by_base.setdefault(base, {})["OKX"] = r.to_dict()

    rows = []
    for base, d in by_base.items():
        k = d.get("KUCOIN")
        o = d.get("OKX")

        if k and o:
            # If directions match -> BOTH row (use better SKOR toward extremes, but keep direction)
            if k["YÖN"] == o["YÖN"]:
                direction = k["YÖN"]
                # choose "stronger" by distance from 50 in RAW
                pick = k if abs(k["RAW"] - 50) >= abs(o["RAW"] - 50) else o
                rows.append(
                    {
                        "YÖN": direction,
                        "COIN": base,
                        "SKOR": int(pick["SKOR"]),
                        "FİYAT": float(pick["FİYAT"]),
                        "RAW": int(pick["RAW"]),
                        "QV_24H": float(max(k["QV_24H"], o["QV_24H"])),
                        "KAPI": int(max(k["KAPI"], o["KAPI"])),  # if either hits 6, show 6
                        "STRONG": bool(k["STRONG"] or o["STRONG"]),
                        "SOURCE": "BOTH",
                    }
                )
            else:
                # Mixed directions -> keep the more extreme one as single source
                pick = k if abs(k["RAW"] - 50) >= abs(o["RAW"] - 50) else o
                rows.append(
                    {
                        "YÖN": pick["YÖN"],
                        "COIN": base,
                        "SKOR": int(pick["SKOR"]),
                        "FİYAT": float(pick["FİYAT"]),
                        "RAW": int(pick["RAW"]),
                        "QV_24H": float(pick["QV_24H"]),
                        "KAPI": int(pick["KAPI"]),
                        "STRONG": bool(pick["STRONG"]),
                        "SOURCE": pick["EX"],
                    }
                )
        elif k:
            rows.append(
                {
                    "YÖN": k["YÖN"],
                    "COIN": base,
                    "SKOR": int(k["SKOR"]),
                    "FİYAT": float(k["FİYAT"]),
                    "RAW": int(k["RAW"]),
                    "QV_24H": float(k["QV_24H"]),
                    "KAPI": int(k["KAPI"]),
                    "STRONG": bool(k["STRONG"]),
                    "SOURCE": "KUCOIN",
                }
            )
        elif o:
            rows.append(
                {
                    "YÖN": o["YÖN"],
                    "COIN": base,
                    "SKOR": int(o["SKOR"]),
                    "FİYAT": float(o["FİYAT"]),
                    "RAW": int(o["RAW"]),
                    "QV_24H": float(o["QV_24H"]),
                    "KAPI": int(o["KAPI"]),
                    "STRONG": bool(o["STRONG"]),
                    "SOURCE": "OKX",
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Priority sorting: BOTH first, then STRONG, then closer to thresholds
    src_rank = df["SOURCE"].map({"BOTH": 0, "KUCOIN": 1, "OKX": 1}).fillna(2).astype(int)
    strong_rank = (~df["STRONG"]).astype(int)  # STRONG first (False=0)
    # closeness metric
    closeness = np.where(df["YÖN"] == "LONG", (STRONG_LONG_MIN - df["SKOR"]).clip(lower=0), (df["SKOR"] - STRONG_SHORT_MAX).clip(lower=0))
    df["_src"] = src_rank
    df["_str"] = strong_rank
    df["_cl"] = closeness

    df = df.sort_values(["_src", "_str", "_cl", "QV_24H"], ascending=[True, True, True, False]).drop(columns=["_src", "_str", "_cl"])
    return df.reset_index(drop=True)


def build_final_table(df_all: pd.DataFrame) -> tuple[pd.DataFrame, dict, str]:
    if df_all is None or df_all.empty:
        return pd.DataFrame(), {"sl": 0, "ss": 0, "l": 0, "s": 0}, "Aday yok (network/KuCoin/OKX veya filtre çok sert)."

    df = df_all.copy()

    # split longs/shorts
    df_long = df[df["YÖN"] == "LONG"].copy()
    df_short = df[df["YÖN"] == "SHORT"].copy()

    # Strong definition (already boolean) but enforce KAPI==6
    df_long["STRONG"] = df_long["STRONG"] & (df_long["KAPI"] == 6) & (df_long["SKOR"] >= STRONG_LONG_MIN)
    df_short["STRONG"] = df_short["STRONG"] & (df_short["KAPI"] == 6) & (df_short["SKOR"] <= STRONG_SHORT_MAX)

    strong_long = df_long[df_long["STRONG"]].copy()
    strong_short = df_short[df_short["STRONG"]].copy()

    # Sort candidates
    strong_long = strong_long.sort_values(["SOURCE", "SKOR", "QV_24H"], ascending=[True, False, False])
    strong_short = strong_short.sort_values(["SOURCE", "SKOR", "QV_24H"], ascending=[True, True, False])

    # Fallback candidates (closest)
    df_long["_cl"] = (STRONG_LONG_MIN - df_long["SKOR"]).abs()
    df_short["_cl"] = (df_short["SKOR"] - STRONG_SHORT_MAX).abs()

    cand_long = df_long.sort_values(["_cl", "SOURCE", "QV_24H"], ascending=[True, True, False]).drop(columns=["_cl"])
    cand_short = df_short.sort_values(["_cl", "SOURCE", "QV_24H"], ascending=[True, True, False]).drop(columns=["_cl"])

    # Build 10+10
    picked = []
    used = set()

    # First strong (BOTH prioritized already via SOURCE sorting in merge; but we still keep)
    for _, r in strong_long.iterrows():
        if len([x for x in picked if x["YÖN"] == "LONG"]) >= 10:
            break
        k = (r["COIN"], r["YÖN"])
        if k in used:
            continue
        used.add(k)
        picked.append(r.to_dict())

    for _, r in strong_short.iterrows():
        if len([x for x in picked if x["YÖN"] == "SHORT"]) >= 10:
            break
        k = (r["COIN"], r["YÖN"])
        if k in used:
            continue
        used.add(k)
        picked.append(r.to_dict())

    # Fill remaining with best candidates
    for _, r in cand_long.iterrows():
        if len([x for x in picked if x["YÖN"] == "LONG"]) >= 10:
            break
        k = (r["COIN"], r["YÖN"])
        if k in used:
            continue
        used.add(k)
        picked.append(r.to_dict())

    for _, r in cand_short.iterrows():
        if len([x for x in picked if x["YÖN"] == "SHORT"]) >= 10:
            break
        k = (r["COIN"], r["YÖN"])
        if k in used:
            continue
        used.add(k)
        picked.append(r.to_dict())

    out = pd.DataFrame(picked)
    if out.empty:
        return pd.DataFrame(), {"sl": 0, "ss": 0, "l": 0, "s": 0}, "Aday yok (network/KuCoin/OKX veya filtre çok sert)."

    # Final sort: BOTH first; inside: STRONG first; then SKOR extreme
    out["_src"] = out["SOURCE"].map({"BOTH": 0, "KUCOIN": 1, "OKX": 1}).fillna(2).astype(int)
    out["_str"] = (~out["STRONG"]).astype(int)
    out["_ord"] = np.where(out["YÖN"] == "LONG", -out["SKOR"], out["SKOR"])
    out = out.sort_values(["_src", "_str", "_ord", "QV_24H"], ascending=[True, True, True, False]).drop(columns=["_src", "_str", "_ord"])

    # counts
    sl = int((out["YÖN"].eq("LONG") & out["STRONG"]).sum())
    ss = int((out["YÖN"].eq("SHORT") & out["STRONG"]).sum())
    l = int((out["YÖN"].eq("LONG")).sum())
    s = int((out["YÖN"].eq("SHORT")).sum())

    msg = "✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu." if (sl + ss) > 0 else "⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu."
    return out.reset_index(drop=True), {"sl": sl, "ss": ss, "l": l, "s": s}, msg


# -----------------------------
# Run scan (with visible progress)
# -----------------------------
progress = st.progress(0, text="Tarama başlıyor…")
status = st.empty()

with st.spinner("KuCoin + OKX taranıyor…"):
    t0 = time.time()

    df_k, meta_k = (pd.DataFrame(), {"universe": 0, "after_liquidity": 0, "scanned": 0})
    df_o, meta_o = (pd.DataFrame(), {"universe": 0, "after_liquidity": 0, "scanned": 0})

    if kucoin_ok:
        status.info("KuCoin taranıyor…")
        df_k, meta_k = scan_exchange("kucoin")
        progress.progress(35, text="KuCoin taraması bitti…")

    if okx_ok:
        status.info("OKX taranıyor…")
        df_o, meta_o = scan_exchange("okx")
        progress.progress(70, text="OKX taraması bitti…")

    status.info("Birleştiriliyor…")
    df_all = merge_kucoin_okx(df_k, df_o)
    df_show, counts, msg = build_final_table(df_all)

    progress.progress(100, text="Tarama bitti ✅")
    status.empty()
    elapsed = max(1, int(time.time() - t0))

# Meta info box
evren_total = int(meta_k.get("universe", 0)) + int(meta_o.get("universe", 0))
after_liq = int(meta_k.get("after_liquidity", 0)) + int(meta_o.get("after_liquidity", 0))
scanned = int(meta_k.get("scanned", 0)) + int(meta_o.get("scanned", 0))

st.info(f"🧠 Evren (USDT spot): {evren_total:,} • Likidite filtresi sonrası: {after_liq:,} • Tarama: {scanned:,} • Süre: {elapsed}s")

# Message + counts
if "✅" in msg:
    st.success(msg)
else:
    st.warning(msg)

st.markdown(
    f"✅ **STRONG LONG:** {counts['sl']}  |  💀 **STRONG SHORT:** {counts['ss']}  |  **LONG:** {counts['l']}  |  **SHORT:** {counts['s']}"
)

st.subheader("🎯 SNIPER TABLO")

if df_show is None or df_show.empty:
    st.error("Aday yok (network/KuCoin/OKX veya filtre çok sert). Bir sonraki auto refresh'i bekle.")
else:
    # Ensure columns order
    cols = ["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI", "STRONG", "SOURCE"]
    df_show = df_show.loc[:, cols].copy()
    st.dataframe(style_table(df_show), use_container_width=True, height=650)
