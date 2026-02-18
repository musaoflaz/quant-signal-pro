from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import ccxt


# ============================================================
# FIXED CONFIG (NO SIDEBAR) — Musa "Auto + sade" modu
# ============================================================
IST_TZ = ZoneInfo("Europe/Istanbul")

TIMEFRAME = "15m"
CANDLE_LIMIT = 200

AUTO_REFRESH_SEC = 240  # 4 dk
TABLE_SIZE = 20

# STRONG thresholds (senin isteğin)
STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10

# Evren / hız dengesi
MAX_SCAN = 900              # pratik limit (KuCoin USDT spot evreninde timeout yememek için)
MIN_QV_24H_USDT = 50_000.0  # çok illiquid çöpleri temizler (spread/boşluk riski)
SLEEP_BETWEEN_REQ = 0.03    # rate-limit dostu

# “Skor adımı”
SCORE_STEP = 5  # 5'er 5'er (daha gerçekçi dağılım)

# 6 Kapı – Seviye 2 (sniper için “zor ama gerçekçi”)
ADX_MIN = 18.0            # trend gücü
ATR_PCT_MIN = 0.45        # hareketlilik (çok sıkışık coinleri ele)
RSI_LONG_MAX = 35.0       # long için oversold
RSI_SHORT_MIN = 65.0      # short için overbought

# ============================================================
# Indicators (pure pandas/numpy)
# ============================================================
def sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p, min_periods=p).mean()

def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False, min_periods=p).mean()

def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def bollinger(series: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)

    atr_s = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_s
    minus_di = 100.0 * pd.Series(minus_dm).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_s

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx_v = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx_v.fillna(0.0)

def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal = ema(macd_line, sig)
    return macd_line - signal

def round_step(x: float, step: int = 5) -> int:
    return int(step * round(float(x) / step))

# ============================================================
# Exchange helpers
# ============================================================
def make_exchange() -> ccxt.kucoin:
    return ccxt.kucoin({"enableRateLimit": True, "timeout": 20000})

@st.cache_data(show_spinner=False, ttl=600)
def load_usdt_spot_symbols() -> list[str]:
    ex = make_exchange()
    markets = ex.load_markets()
    out = []
    for sym, m in markets.items():
        if not m:
            continue
        if not m.get("active", True):
            continue
        if not m.get("spot", False):
            continue
        if m.get("quote") != "USDT":
            continue
        out.append(sym)
    return sorted(set(out))

def safe_fetch_tickers(ex: ccxt.Exchange, symbols: list[str]) -> dict:
    # KuCoin bazen toplu/tekil fark ediyor. Güvenli yol:
    try:
        return ex.fetch_tickers(symbols)
    except Exception:
        try:
            all_t = ex.fetch_tickers()
            return {s: all_t.get(s) for s in symbols if s in all_t}
        except Exception:
            return {}

def qv_usdt(t: dict) -> float:
    if not t or not isinstance(t, dict):
        return 0.0
    v = t.get("quoteVolume")
    if v is not None:
        try:
            return float(v)
        except Exception:
            return 0.0
    # fallback: baseVolume * last
    try:
        bv = float(t.get("baseVolume") or 0.0)
        last = float(t.get("last") or 0.0)
        return bv * last
    except Exception:
        return 0.0

def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

# ============================================================
# “6 Kapı / Seviye 2” + RAW/Score
# ============================================================
def compute_raw_and_gates(df: pd.DataFrame) -> tuple[float, int, str]:
    """
    Returns:
      raw (0..100 bullishness),
      gate_count (0..6),
      direction ("LONG" if raw>=50 else "SHORT")
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    rsi14 = rsi_wilder(close, 14)
    _, bb_up, bb_low = bollinger(close, 20, 2.0)

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)

    hist = macd_hist(close, 12, 26, 9)
    adx14 = adx(high, low, close, 14)
    atr14 = atr(high, low, close, 14)
    atr_pct = (atr14 / close.replace(0.0, np.nan)) * 100.0

    last = float(close.iloc[-1])
    last_rsi = float(rsi14.iloc[-1])
    last_bb_low = float(bb_low.iloc[-1])
    last_bb_up = float(bb_up.iloc[-1])
    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_ema200 = float(ema200.iloc[-1])
    last_hist = float(hist.iloc[-1])
    prev_hist = float(hist.iloc[-2])
    last_adx = float(adx14.iloc[-1])
    last_atr_pct = float(atr_pct.iloc[-1])

    if any(np.isnan([last_rsi, last_bb_low, last_bb_up, last_ema20, last_ema50, last_ema200, last_hist, prev_hist, last_adx, last_atr_pct])):
        return 50.0, 0, "LONG"

    # ------------------------
    # RAW (bullishness) 0..100
    # ------------------------
    raw = 50.0

    # Trend bias (ema50 vs ema200)
    if last_ema50 > last_ema200:
        raw += 10
    elif last_ema50 < last_ema200:
        raw -= 10

    # Pullback / extension via BB
    if last <= last_bb_low:
        raw += 15
    elif last >= last_bb_up:
        raw -= 15

    # RSI extremes (sniper)
    if last_rsi <= 30:
        raw += 15
    elif last_rsi <= RSI_LONG_MAX:
        raw += 10
    elif last_rsi >= 70:
        raw -= 15
    elif last_rsi >= RSI_SHORT_MIN:
        raw -= 10

    # MACD histogram slope (micro reversal)
    if last_hist > prev_hist:
        raw += 5
    else:
        raw -= 5

    # ADX confirms trend strength (align with trend)
    if last_adx >= ADX_MIN:
        if last_ema50 > last_ema200:
            raw += 5
        elif last_ema50 < last_ema200:
            raw -= 5

    # ATR%: very low movement => pull to neutral (avoid fake 100/0)
    if last_atr_pct < ATR_PCT_MIN:
        raw = 50 + (raw - 50) * 0.35  # soften extremes

    raw = float(np.clip(raw, 0, 100))
    direction = "LONG" if raw >= 50 else "SHORT"

    # ------------------------
    # 6 Kapı (Seviye 2)
    # ------------------------
    gates = 0

    # Gate 1: Trend regime alignment
    trend_up = last_ema50 > last_ema200
    trend_dn = last_ema50 < last_ema200
    if (direction == "LONG" and trend_up) or (direction == "SHORT" and trend_dn):
        gates += 1

    # Gate 2: ADX strength
    if last_adx >= ADX_MIN:
        gates += 1

    # Gate 3: ATR% enough movement
    if last_atr_pct >= ATR_PCT_MIN:
        gates += 1

    # Gate 4: Pullback to BB extreme (sniper entry zone)
    if direction == "LONG" and last <= last_bb_low:
        gates += 1
    if direction == "SHORT" and last >= last_bb_up:
        gates += 1

    # Gate 5: RSI extreme in the correct side
    if direction == "LONG" and last_rsi <= RSI_LONG_MAX:
        gates += 1
    if direction == "SHORT" and last_rsi >= RSI_SHORT_MIN:
        gates += 1

    # Gate 6: MACD histogram turns toward reversal
    if direction == "LONG" and last_hist > prev_hist:
        gates += 1
    if direction == "SHORT" and last_hist < prev_hist:
        gates += 1

    return raw, int(gates), direction

# ============================================================
# Table selection logic
# ============================================================
def build_table(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all

    df = df_all.copy()

    # Score step quantization
    df["SKOR"] = df["RAW"].apply(lambda x: round_step(x, SCORE_STEP)).astype(int)

    # direction from RAW (simple & consistent)
    df["YÖN"] = np.where(df["RAW"] >= 50, "LONG", "SHORT")

    # STRONG flags (both sides)
    df["IS_STRONG_LONG"] = df["SKOR"] >= STRONG_LONG_MIN
    df["IS_STRONG_SHORT"] = df["SKOR"] <= STRONG_SHORT_MAX
    df["IS_STRONG"] = df["IS_STRONG_LONG"] | df["IS_STRONG_SHORT"]

    # Severity for sorting:
    # long severity: higher score
    # short severity: lower score
    df["SEVERITY"] = np.where(df["YÖN"] == "LONG", df["SKOR"], 100 - df["SKOR"])

    # Split
    strong = df[df["IS_STRONG"]].copy()
    long_candidates = df[df["YÖN"] == "LONG"].copy()
    short_candidates = df[df["YÖN"] == "SHORT"].copy()

    # Sort
    strong = strong.sort_values(["SEVERITY", "QV_24H"], ascending=[False, False])
    long_candidates = long_candidates.sort_values(["SKOR", "QV_24H"], ascending=[False, False])
    short_candidates = short_candidates.sort_values(["SKOR", "QV_24H"], ascending=[True, False])

    # Start table with strong
    out_rows = []
    used = set()

    def push_row(r):
        k = str(r["COIN"])
        if k in used:
            return
        out_rows.append(r)
        used.add(k)

    for _, r in strong.head(TABLE_SIZE).iterrows():
        push_row(r)

    # Fill remaining with TOP candidates (LONG+SHORT birlikte)
    remaining = TABLE_SIZE - len(out_rows)
    if remaining > 0:
        # Make balanced fill so SHORT kaybolmasın
        long_iter = long_candidates.iterrows()
        short_iter = short_candidates.iterrows()

        # current counts
        cur_long = sum(1 for x in out_rows if x["YÖN"] == "LONG")
        cur_short = sum(1 for x in out_rows if x["YÖN"] == "SHORT")

        # alternate picking the side that is currently fewer
        while remaining > 0:
            pick_long = (cur_long <= cur_short)

            picked = False
            if pick_long:
                for _, r in long_iter:
                    if str(r["COIN"]) in used:
                        continue
                    push_row(r); cur_long += 1; remaining -= 1; picked = True
                    break
                if not picked:
                    for _, r in short_iter:
                        if str(r["COIN"]) in used:
                            continue
                        push_row(r); cur_short += 1; remaining -= 1; picked = True
                        break
            else:
                for _, r in short_iter:
                    if str(r["COIN"]) in used:
                        continue
                    push_row(r); cur_short += 1; remaining -= 1; picked = True
                    break
                if not picked:
                    for _, r in long_iter:
                        if str(r["COIN"]) in used:
                            continue
                        push_row(r); cur_long += 1; remaining -= 1; picked = True
                        break

            if not picked:
                break

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out

    # Final columns (order)
    out = out[["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI"]].copy()

    # Sort inside table: STRONG first, then best TOP
    out["_STR"] = np.where((out["SKOR"] >= STRONG_LONG_MIN) | (out["SKOR"] <= STRONG_SHORT_MAX), 1, 0)
    out["_SEV"] = np.where(out["YÖN"] == "LONG", out["SKOR"], 100 - out["SKOR"])
    out = out.sort_values(["_STR", "_SEV", "QV_24H"], ascending=[False, False, False]).drop(columns=["_STR", "_SEV"])
    out = out.reset_index(drop=True)
    return out

# ============================================================
# Styling (Dark theme + readable table)
# ============================================================
def style_table(df: pd.DataFrame):
    def dir_style(v):
        if v == "LONG":
            return "background-color:#0f3d2e;color:#e6edf3;font-weight:700;"
        if v == "SHORT":
            return "background-color:#4a1414;color:#e6edf3;font-weight:700;"
        return ""

    def score_style(v):
        try:
            v = int(v)
        except Exception:
            return ""
        if v >= STRONG_LONG_MIN:
            return "background-color:#0b5d1e;color:#ffffff;font-weight:800;"
        if v <= STRONG_SHORT_MAX:
            return "background-color:#7a1111;color:#ffffff;font-weight:800;"
        return "background-color:#0b1220;color:#e6edf3;font-weight:700;"

    def raw_style(v):
        try:
            v = float(v)
        except Exception:
            return ""
        # show raw intensity subtly
        if v >= 80:
            return "background-color:#083b15;color:#e6edf3;"
        if v <= 20:
            return "background-color:#3b0b0b;color:#e6edf3;"
        return "background-color:#0b1220;color:#e6edf3;"

    fmt = {
        "FİYAT": "{:.6f}",
        "RAW": "{:.0f}",
        "QV_24H": "{:,.0f}",
        "KAPI": "{:.0f}",
        "SKOR": "{:.0f}",
    }

    return (
        df.style
        .format(fmt)
        .applymap(dir_style, subset=["YÖN"])
        .applymap(score_style, subset=["SKOR"])
        .applymap(raw_style, subset=["RAW"])
        .set_properties(**{
            "background-color": "#0b1220",
            "color": "#e6edf3",
            "border-color": "#1f2a37",
            "font-size": "14px",
        })
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#0b0f14"), ("color", "#e6edf3"), ("border-color", "#1f2a37"), ("font-weight", "800")]},
            {"selector": "td", "props": [("border-color", "#1f2a37")]},
            {"selector": "table", "props": [("border-collapse", "collapse")]},
        ])
    )

# ============================================================
# Streamlit page
# ============================================================
st.set_page_config(page_title="KuCoin PRO Sniper — Auto (LONG+SHORT)", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.0rem; }
h1, h2, h3, h4, h5, h6, p, span, div { color: #e6edf3 !important; }
[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
[data-testid="stToolbar"] { display:none; }
a { color: #93c5fd !important; }
</style>
""",
    unsafe_allow_html=True,
)

# Auto refresh
try:
    st.autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
except Exception:
    try:
        st.experimental_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
    except Exception:
        pass

# Header
now_ist = datetime.now(IST_TZ)
st.markdown(
    f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;">
  <div>
    <div style="font-size:32px;font-weight:900;letter-spacing:0.2px;">🎯 KuCoin PRO Sniper — Auto (LONG + SHORT)</div>
    <div style="opacity:0.85;margin-top:6px;">TF={TIMEFRAME} • STRONG: SKOR≥{STRONG_LONG_MIN} (LONG) / SKOR≤{STRONG_SHORT_MAX} (SHORT) • 6 Kapı • Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s</div>
  </div>
  <div style="text-align:right;opacity:0.9;">
    <div style="font-size:12px;">Istanbul Time</div>
    <div style="font-size:18px;font-weight:800;">{now_ist.strftime('%Y-%m-%d %H:%M:%S')}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# Scan universe
ex = make_exchange()

with st.spinner("⏳ KuCoin USDT spot evreni taranıyor... (lütfen bekle)"):
    symbols = load_usdt_spot_symbols()

    tickers = safe_fetch_tickers(ex, symbols)
    rows_rank = []
    for s in symbols:
        t = tickers.get(s)
        qv = qv_usdt(t)
        if qv >= MIN_QV_24H_USDT:
            rows_rank.append((s, qv))
    rows_rank.sort(key=lambda x: x[1], reverse=True)

    # Scan list (liquidity filtered then capped for stability)
    scan_list = [s for s, _ in rows_rank[:MAX_SCAN]]

    st.info(f"🧠 Evren (USDT spot): {len(symbols)} • Likidite filtresi sonrası: {len(rows_rank)} • Tarama: {len(scan_list)}")

    prog = st.progress(0, text="Scanning...")
    out_rows = []

    total = max(1, len(scan_list))
    for i, sym in enumerate(scan_list, start=1):
        try:
            ohlcv = safe_fetch_ohlcv(ex, sym, TIMEFRAME, CANDLE_LIMIT)
            if not ohlcv or len(ohlcv) < 210:
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            raw, gates, direction = compute_raw_and_gates(df)

            last_price = float(df["close"].iloc[-1])
            qv24 = float(dict(rows_rank).get(sym, 0.0))

            out_rows.append({
                "COIN": sym.split("/")[0],
                "FİYAT": last_price,
                "RAW": raw,
                "KAPI": gates,
                "QV_24H": qv24,
            })

        except (ccxt.RequestTimeout, ccxt.NetworkError):
            pass
        except ccxt.ExchangeError:
            pass
        except Exception:
            pass

        if i % 5 == 0:
            prog.progress(int((i / total) * 100), text=f"{i}/{total} tarandı...")
        time.sleep(SLEEP_BETWEEN_REQ)

    prog.progress(100, text="Tarama bitti ✅")

df_all = pd.DataFrame(out_rows)

st.write("")
st.markdown("## 🎯 SNIPER TABLO")

if df_all.empty:
    st.warning("Aday yok (network/KuCoin veya filtre çok sert). Bir sonraki auto refresh’i bekle.")
else:
    table = build_table(df_all)

    # Status banner
    strong_count = int(((table["SKOR"] >= STRONG_LONG_MIN) | (table["SKOR"] <= STRONG_SHORT_MAX)).sum())
    if strong_count > 0:
        st.success("✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu.")
    else:
        st.warning("⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu (LONG+SHORT birlikte).")

    st.dataframe(style_table(table), use_container_width=True, height=720)
