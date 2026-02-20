# requirements.txt:
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
# FINAL BASE (AUTO, NO SIDEBAR)
# =============================
IST_TZ = ZoneInfo("Europe/Istanbul")

TF_LTF = "15m"
TF_HTF = "1h"
AUTO_REFRESH_SEC = 240  # 4 dk

TOP_N_PER_EXCHANGE = 150
TABLE_ROWS = 20
CANDLE_LIMIT = 200

RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIOD = 20
ADX_PERIOD = 14

# STRONG thresholds
STRONG_LONG_MIN = 90   # RAW >= 90
STRONG_SHORT_MAX = 10  # RAW <= 10

SCORE_STEP = 5  # SKOR 5'lik

# 6 Kapı (Level 2)
GATE_ADX_MIN = 18.0
GATE_SPREAD_MAX = 0.004  # 0.40%
GATE_QV_MIN = 0.0        # küçük coinler de gelsin (senin isteğin)

GATE_RSI_LONG_MAX = 35.0
GATE_RSI_SHORT_MIN = 65.0


# =============================
# Indicators (pure pandas/numpy)
# =============================
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def bollinger_bands(series: pd.Series, period: int, n_std: float):
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

    tr_s = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_dm_s = pd.Series(plus_dm, index=high.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_s = pd.Series(minus_dm, index=high.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (plus_dm_s / tr_s.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm_s / tr_s.replace(0.0, np.nan))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)


def round_step(x: float, step: int) -> int:
    if step <= 1:
        return int(round(x))
    return int(np.clip(int(round(x / step) * step), 0, 100))


# =============================
# Exchanges
# =============================
def make_exchange(name: str) -> ccxt.Exchange:
    if name == "KUCOIN":
        return ccxt.kucoin({"enableRateLimit": True, "timeout": 20000})
    if name == "OKX":
        return ccxt.okx({"enableRateLimit": True, "timeout": 20000})
    raise ValueError("Unknown exchange")


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


def list_usdt_spot_symbols(markets: dict) -> list[str]:
    syms = []
    for sym, m in (markets or {}).items():
        try:
            if not m:
                continue
            if not m.get("active", True):
                continue
            if not m.get("spot", False):
                continue
            if m.get("quote") != "USDT":
                continue
            syms.append(sym)
        except Exception:
            continue
    return sorted(set(syms))


def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def compute_qv_24h(t: dict) -> float:
    if not t or not isinstance(t, dict):
        return 0.0
    qv = t.get("quoteVolume")
    if qv is not None:
        try:
            return float(qv)
        except Exception:
            return 0.0
    try:
        bv = float(t.get("baseVolume") or 0.0)
        last = float(t.get("last") or 0.0)
        return bv * last
    except Exception:
        return 0.0


def compute_spread_ratio(t: dict) -> float:
    if not t or not isinstance(t, dict):
        return 1.0
    try:
        bid = float(t.get("bid") or 0.0)
        ask = float(t.get("ask") or 0.0)
        if bid <= 0 or ask <= 0:
            return 1.0
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return 1.0
        return (ask - bid) / mid
    except Exception:
        return 1.0


# =============================
# Scoring (LONG + SHORT ayrı)
# =============================
def raw_score_long(close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float) -> float:
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
    return float(np.clip(score, 0.0, 100.0))


def raw_strength_short(close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float) -> float:
    # HIGH => strong short strength
    score = 50.0
    score += 20.0 if close < sma20_v else -20.0
    if rsi_v > 65.0:
        score += 40.0
    elif rsi_v < 35.0:
        score -= 40.0
    if close >= bb_up:
        score += 40.0
    elif close <= bb_low:
        score -= 40.0
    return float(np.clip(score, 0.0, 100.0))


def make_rows_for_symbol(ex_name: str, symbol: str, ticker: dict, ohlcv_ltf: list, ohlcv_htf: list) -> list[dict]:
    if not ohlcv_ltf or not ohlcv_htf:
        return []

    df = pd.DataFrame(ohlcv_ltf, columns=["ts", "open", "high", "low", "close", "vol"])
    dfh = pd.DataFrame(ohlcv_htf, columns=["ts", "open", "high", "low", "close", "vol"])
    if df.empty or dfh.empty:
        return []

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close_h = dfh["close"].astype(float)

    sma20 = sma(close, SMA_PERIOD)
    _, bb_up, bb_low = bollinger_bands(close, BB_PERIOD, BB_STD)
    rsi14 = rsi_wilder(close, RSI_PERIOD)
    adx14 = adx_wilder(high, low, close, ADX_PERIOD)

    sma20_h = sma(close_h, SMA_PERIOD)

    last_close = float(close.iloc[-1])
    last_sma20 = float(sma20.iloc[-1])
    last_rsi = float(rsi14.iloc[-1])
    last_low = float(bb_low.iloc[-1])
    last_up = float(bb_up.iloc[-1])
    last_adx = float(adx14.iloc[-1])

    last_sma20_h = float(sma20_h.iloc[-1])
    last_close_h = float(close_h.iloc[-1])

    if any(np.isnan([last_sma20, last_rsi, last_low, last_up, last_adx, last_sma20_h, last_close_h])):
        return []

    qv = compute_qv_24h(ticker)
    spr = compute_spread_ratio(ticker)

    last_price = float(ticker.get("last") or last_close)
    coin = symbol.split("/")[0].strip().upper()

    # RAW values
    raw_l = raw_score_long(last_close, last_sma20, last_rsi, last_low, last_up)
    strength_s = raw_strength_short(last_close, last_sma20, last_rsi, last_low, last_up)
    raw_s_display = 100.0 - strength_s  # small => strong short

    # Gates shared
    g_qv = qv >= GATE_QV_MIN
    g_spread = spr <= GATE_SPREAD_MAX
    g_adx = last_adx >= GATE_ADX_MIN

    # HTF align
    g_htf_long = last_close_h > last_sma20_h
    g_htf_short = last_close_h < last_sma20_h

    # RSI extreme
    g_rsi_long = last_rsi <= GATE_RSI_LONG_MAX
    g_rsi_short = last_rsi >= GATE_RSI_SHORT_MIN

    # BB touch
    g_bb_long = last_close <= last_low
    g_bb_short = last_close >= last_up

    kapi_long = int(g_qv) + int(g_spread) + int(g_adx) + int(g_htf_long) + int(g_rsi_long) + int(g_bb_long)
    kapi_short = int(g_qv) + int(g_spread) + int(g_adx) + int(g_htf_short) + int(g_rsi_short) + int(g_bb_short)

    strong_long = (kapi_long == 6) and (raw_l >= STRONG_LONG_MIN)
    strong_short = (kapi_short == 6) and (raw_s_display <= STRONG_SHORT_MAX)

    return [
        {
            "YÖN": "LONG",
            "COIN": coin,
            "SKOR": round_step(raw_l, SCORE_STEP),
            "FİYAT": float(last_price),
            "RAW": int(round(raw_l)),
            "QV_24H": float(qv),
            "KAPI": int(kapi_long),
            "STRONG": bool(strong_long),
            "SOURCE": ex_name,
        },
        {
            "YÖN": "SHORT",
            "COIN": coin,
            "SKOR": round_step(raw_s_display, SCORE_STEP),
            "FİYAT": float(last_price),
            "RAW": int(round(raw_s_display)),
            "QV_24H": float(qv),
            "KAPI": int(kapi_short),
            "STRONG": bool(strong_short),
            "SOURCE": ex_name,
        },
    ]


def strength_key(row: dict) -> float:
    raw = float(row.get("RAW", 50))
    if row.get("YÖN") == "SHORT":
        return 100.0 - raw  # raw small => stronger short
    return raw


# =============================
# Source merge per COIN + DIRECTION
# (fix: keep LONG and SHORT separately)
# =============================
def merge_sources_per_direction(rows_ku: list[dict], rows_okx: list[dict]) -> list[dict]:
    # key: (COIN, YÖN) -> best row among KUCOIN/OKX, but mark BOTH if coin+dir exists on both
    by_key: dict[tuple[str, str], dict] = {}
    seen_sources: dict[tuple[str, str], set[str]] = {}

    for r in rows_ku + rows_okx:
        key = (r["COIN"], r["YÖN"])
        seen_sources.setdefault(key, set()).add(r["SOURCE"])

        if key not in by_key or strength_key(r) > strength_key(by_key[key]):
            by_key[key] = r

    out = []
    for key, best in by_key.items():
        sset = seen_sources.get(key, set())
        best = dict(best)
        if "KUCOIN" in sset and "OKX" in sset:
            best["SOURCE"] = "BOTH"
        out.append(best)

    return out


# =============================
# Table pick (fix: ensure SHORTs can show)
# Priority: BOTH > STRONG > KAPI > strength > QV
# Fill with best overall; if one side empty, naturally only that side comes.
# =============================
def pick_table(rows: list[dict], n: int) -> list[dict]:
    def pri(r: dict) -> tuple:
        both = 1 if r.get("SOURCE") == "BOTH" else 0
        strong = 1 if r.get("STRONG") else 0
        kapi = int(r.get("KAPI", 0))
        s = float(strength_key(r))
        qv = float(r.get("QV_24H", 0.0))
        # higher better
        return (both, strong, kapi, s, qv)

    rows_sorted = sorted(rows, key=pri, reverse=True)

    # If we can, keep some balance (not forced): try to place BOTH-first, then alternate sides
    picked = []
    used = set()

    # 1) take all BOTH strong first
    for r in rows_sorted:
        key = (r["COIN"], r["YÖN"])
        if key in used:
            continue
        if r.get("SOURCE") == "BOTH" and r.get("STRONG"):
            picked.append(r)
            used.add(key)
            if len(picked) >= n:
                return picked

    # 2) take remaining STRONG (any source)
    for r in rows_sorted:
        key = (r["COIN"], r["YÖN"])
        if key in used:
            continue
        if r.get("STRONG"):
            picked.append(r)
            used.add(key)
            if len(picked) >= n:
                return picked

    # 3) fill remaining with best candidates, but alternate LONG/SHORT when possible
    longs = [r for r in rows_sorted if r.get("YÖN") == "LONG"]
    shorts = [r for r in rows_sorted if r.get("YÖN") == "SHORT"]

    i = j = 0
    want_long = True
    while len(picked) < n and (i < len(longs) or j < len(shorts)):
        if want_long and i < len(longs):
            r = longs[i]; i += 1
            key = (r["COIN"], r["YÖN"])
            if key not in used:
                picked.append(r); used.add(key)
        elif (not want_long) and j < len(shorts):
            r = shorts[j]; j += 1
            key = (r["COIN"], r["YÖN"])
            if key not in used:
                picked.append(r); used.add(key)
        else:
            # fallback: whichever side still has items
            if i < len(longs):
                r = longs[i]; i += 1
                key = (r["COIN"], r["YÖN"])
                if key not in used:
                    picked.append(r); used.add(key)
            elif j < len(shorts):
                r = shorts[j]; j += 1
                key = (r["COIN"], r["YÖN"])
                if key not in used:
                    picked.append(r); used.add(key)

        want_long = not want_long

    return picked[:n]


# =============================
# Styling (LONG green / SHORT red)
# STRONG darker; BOTH extra-dark + subtle border
# =============================
def style_table(df: pd.DataFrame):
    def row_style(row):
        yon = str(row.get("YÖN", ""))
        strong = bool(row.get("STRONG", False))
        source = str(row.get("SOURCE", ""))

        if yon == "LONG":
            base = "#0b3b2f"
            strong_bg = "#06402b"
            both_bg = "#04281b"
        else:
            base = "#3b0b0b"
            strong_bg = "#4a0a0a"
            both_bg = "#2a0505"

        bg = base
        if strong:
            bg = strong_bg
        if source == "BOTH":
            bg = both_bg  # BOTH daha koyu

        border = "1px solid #1f2a37"
        if source == "BOTH":
            border = "1px solid #7c3aed"  # hafif vurgu

        return [f"background-color: {bg}; color: #e6edf3; {border};"] * len(row)

    fmt = {
        "SKOR": "{:.0f}",
        "FİYAT": "{:.6f}",
        "RAW": "{:.0f}",
        "QV_24H": "{:,.0f}",
        "KAPI": "{:.0f}",
    }
    return df.style.format(fmt).apply(row_style, axis=1)


# =============================
# Streamlit UI (AUTO)
# =============================
st.set_page_config(page_title="KuCoin PRO Sniper — Auto (LONG + SHORT)", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }
h1, h2, h3, h4, h5, h6, p, span, div { color: #e6edf3 !important; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
</style>
""",
    unsafe_allow_html=True,
)

# Auto refresh
try:
    st.autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh_main")
except Exception:
    try:
        st.experimental_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh_main")
    except Exception:
        pass

st.title("🎯 KuCoin PRO Sniper — Auto (LONG + SHORT)")
st.caption(
    f"TF={TF_LTF} • HTF={TF_HTF} • STRONG: SKOR≥{STRONG_LONG_MIN} (LONG) / SKOR≤{STRONG_SHORT_MAX} (SHORT) • 6 Kapı • Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s"
)

st.markdown(f"**İstanbul Time:** {datetime.now(IST_TZ).strftime('%Y-%m-%d %H:%M:%S')}")

# Status boxes
c1, c2 = st.columns(2)
ku_status = c1.empty()
okx_status = c2.empty()

# Progress
progress = st.progress(0, text="Hazırlanıyor…")
scan_box = st.empty()

rows_ku: list[dict] = []
rows_okx: list[dict] = []


def scan_exchange(ex_name: str) -> tuple[bool, str, list[dict]]:
    ex = make_exchange(ex_name)
    markets = safe_load_markets(ex)
    if not markets:
        return False, "load_markets başarısız", []

    symbols = list_usdt_spot_symbols(markets)
    if not symbols:
        return False, "USDT spot sembol yok", []

    tickers = safe_fetch_tickers(ex, symbols)

    ranked = []
    for s in symbols:
        qv = compute_qv_24h(tickers.get(s) or {})
        ranked.append((s, qv))
    ranked.sort(key=lambda x: x[1], reverse=True)

    top_syms = [s for s, _ in ranked[:TOP_N_PER_EXCHANGE]]
    out: list[dict] = []
    total = len(top_syms)

    for i, sym in enumerate(top_syms, start=1):
        scan_box.info(f"Taranıyor ({ex_name}): {i}/{total} • {sym}")
        try:
            t = tickers.get(sym) or {}
            ohlcv_ltf = safe_fetch_ohlcv(ex, sym, TF_LTF, CANDLE_LIMIT)
            ohlcv_htf = safe_fetch_ohlcv(ex, sym, TF_HTF, CANDLE_LIMIT)
            if not ohlcv_ltf or not ohlcv_htf:
                continue
            if len(ohlcv_ltf) < 60 or len(ohlcv_htf) < 60:
                continue

            out.extend(make_rows_for_symbol(ex_name, sym, t, ohlcv_ltf, ohlcv_htf))
        except (ccxt.NetworkError, ccxt.RequestTimeout):
            pass
        except Exception:
            pass

        # pacing
        time.sleep(0.02)

    return True, "Bağlandı", out


with st.spinner("⏳ KuCoin + OKX taranıyor…"):
    # KuCoin
    try:
        ku_ok, ku_msg, rows_ku = scan_exchange("KUCOIN")
    except Exception as e:
        ku_ok, ku_msg, rows_ku = False, f"{type(e).__name__}: {e}", []

    # OKX
    try:
        okx_ok, okx_msg, rows_okx = scan_exchange("OKX")
    except Exception as e:
        okx_ok, okx_msg, rows_okx = False, f"{type(e).__name__}: {e}", []

# Status render
ku_status.success("KuCoin: ✅ Bağlandı" if ku_ok else f"KuCoin: ❌ Hata • {ku_msg}")
okx_status.success("OKX: ✅ Bağlandı" if okx_ok else f"OKX: ❌ Hata • {okx_msg}")

progress.progress(35, text="Kaynaklar birleştiriliyor…")

# Merge per direction (FIX)
merged = merge_sources_per_direction(rows_ku, rows_okx)

progress.progress(70, text="Tablo seçimi yapılıyor…")

picked = pick_table(merged, TABLE_ROWS)

progress.progress(100, text="Tarama bitti ✅")
scan_box.empty()

if not picked:
    st.warning("Aday yok (network / borsa / filtre çok sert). Bir sonraki auto refresh’i bekle.")
    st.stop()

df = pd.DataFrame(picked)

# Counts (only STRONG = KAPI==6 + threshold)
strong_long = int(((df["YÖN"] == "LONG") & (df["STRONG"] == True)).sum())
strong_short = int(((df["YÖN"] == "SHORT") & (df["STRONG"] == True)).sum())
longs = int((df["YÖN"] == "LONG").sum())
shorts = int((df["YÖN"] == "SHORT").sum())

any_strong = bool(df["STRONG"].any())
if any_strong:
    st.success("✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu.")
else:
    st.warning("⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu.")

st.info(f"✅ STRONG LONG: {strong_long} | 💀 STRONG SHORT: {strong_short} | LONG: {longs} | SHORT: {shorts}")

df_show = df[["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI", "STRONG", "SOURCE"]].copy()
df_show["SKOR"] = pd.to_numeric(df_show["SKOR"], errors="coerce").fillna(0).astype(int)
df_show["RAW"] = pd.to_numeric(df_show["RAW"], errors="coerce").fillna(0).astype(int)
df_show["KAPI"] = pd.to_numeric(df_show["KAPI"], errors="coerce").fillna(0).astype(int)
df_show["QV_24H"] = pd.to_numeric(df_show["QV_24H"], errors="coerce").fillna(0.0).astype(float)
df_show["FİYAT"] = pd.to_numeric(df_show["FİYAT"], errors="coerce").fillna(0.0).astype(float)

# Sort display: BOTH on top, then strong, then score strength
def disp_pri(r):
    both = 1 if r["SOURCE"] == "BOTH" else 0
    strong = 1 if bool(r["STRONG"]) else 0
    kapi = int(r["KAPI"])
    raw = int(r["RAW"])
    # strength in display sort
    strength = (100 - raw) if r["YÖN"] == "SHORT" else raw
    return (both, strong, kapi, strength, float(r["QV_24H"]))

df_show = df_show.sort_values(
    by=["SOURCE", "STRONG", "KAPI", "RAW", "QV_24H"],
    ascending=[False, False, False, False, False],
).reset_index(drop=True)

st.subheader("🎯 SNIPER TABLO")
st.dataframe(style_table(df_show), use_container_width=True, height=720)
