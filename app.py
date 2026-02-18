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

AUTO_REFRESH_SEC = 240
TABLE_SIZE = 20

# STRONG thresholds (senin isteğin)
STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10

# Evren / hız dengesi
MAX_SCAN = 450               # 900 yerine 450 (stabil + hızlı, senin ekranda da 425-450 görünüyor)
MIN_QV_24H_USDT = 50_000.0   # illiquid çöpleri temizler
SLEEP_BETWEEN_REQ = 0.03

# Skor adımı (5'er 5'er)
SCORE_STEP = 5

# 6 Kapı – Seviye 2
ADX_MIN = 18.0
ATR_PCT_MIN = 0.45
RSI_LONG_MAX = 35.0
RSI_SHORT_MIN = 65.0


# ============================================================
# Indicators (pure pandas/numpy)
# ============================================================
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
    try:
        bv = float(t.get("baseVolume") or 0.0)
        last = float(t.get("last") or 0.0)
        return bv * last
    except Exception:
        return 0.0


# ============================================================
# “6 Kapı / Seviye 2” + RAW
# ============================================================
def compute_raw_and_gates(df: pd.DataFrame) -> tuple[float, int]:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    rsi14 = rsi_wilder(close, 14)
    _, bb_up, bb_low = bollinger(close, 20, 2.0)

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
    last_ema50 = float(ema50.iloc[-1])
    last_ema200 = float(ema200.iloc[-1])
    last_hist = float(hist.iloc[-1])
    prev_hist = float(hist.iloc[-2])
    last_adx = float(adx14.iloc[-1])
    last_atr_pct = float(atr_pct.iloc[-1])

    if any(np.isnan([last_rsi, last_bb_low, last_bb_up, last_ema50, last_ema200, last_hist, prev_hist, last_adx, last_atr_pct])):
        return 50.0, 0

    # RAW bullishness 0..100
    raw = 50.0

    trend_up = last_ema50 > last_ema200
    trend_dn = last_ema50 < last_ema200

    if trend_up:
        raw += 10
    elif trend_dn:
        raw -= 10

    if last <= last_bb_low:
        raw += 15
    elif last >= last_bb_up:
        raw -= 15

    if last_rsi <= 30:
        raw += 15
    elif last_rsi <= RSI_LONG_MAX:
        raw += 10
    elif last_rsi >= 70:
        raw -= 15
    elif last_rsi >= RSI_SHORT_MIN:
        raw -= 10

    if last_hist > prev_hist:
        raw += 5
    else:
        raw -= 5

    if last_adx >= ADX_MIN:
        raw += 5 if trend_up else (-5 if trend_dn else 0)

    if last_atr_pct < ATR_PCT_MIN:
        raw = 50 + (raw - 50) * 0.35

    raw = float(np.clip(raw, 0, 100))
    direction_long = raw >= 50

    # 6 Kapı
    gates = 0

    # Gate1 trend align
    if (direction_long and trend_up) or ((not direction_long) and trend_dn):
        gates += 1
    # Gate2 ADX
    if last_adx >= ADX_MIN:
        gates += 1
    # Gate3 ATR%
    if last_atr_pct >= ATR_PCT_MIN:
        gates += 1
    # Gate4 BB extreme
    if direction_long and last <= last_bb_low:
        gates += 1
    if (not direction_long) and last >= last_bb_up:
        gates += 1
    # Gate5 RSI extreme
    if direction_long and last_rsi <= RSI_LONG_MAX:
        gates += 1
    if (not direction_long) and last_rsi >= RSI_SHORT_MIN:
        gates += 1
    # Gate6 MACD turn
    if direction_long and last_hist > prev_hist:
        gates += 1
    if (not direction_long) and last_hist < prev_hist:
        gates += 1

    return raw, int(gates)


# ============================================================
# Table logic
# ============================================================
def build_table(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()

    df["SKOR"] = df["RAW"].apply(lambda x: round_step(x, SCORE_STEP)).astype(int)
    df["YÖN"] = np.where(df["RAW"] >= 50, "LONG", "SHORT")

    df["IS_STRONG_LONG"] = df["SKOR"] >= STRONG_LONG_MIN
    df["IS_STRONG_SHORT"] = df["SKOR"] <= STRONG_SHORT_MAX
    df["IS_STRONG"] = df["IS_STRONG_LONG"] | df["IS_STRONG_SHORT"]

    df["SEVERITY"] = np.where(df["YÖN"] == "LONG", df["SKOR"], 100 - df["SKOR"])

    strong = df[df["IS_STRONG"]].sort_values(["SEVERITY", "QV_24H"], ascending=[False, False]).copy()
    longs = df[df["YÖN"] == "LONG"].sort_values(["SKOR", "QV_24H"], ascending=[False, False]).copy()
    shorts = df[df["YÖN"] == "SHORT"].sort_values(["SKOR", "QV_24H"], ascending=[True, False]).copy()

    out = []
    used = set()

    def push_row(r):
        c = str(r["COIN"])
        if c in used:
            return False
        used.add(c)
        out.append(r)
        return True

    # 1) STRONG önce
    for _, r in strong.head(TABLE_SIZE).iterrows():
        push_row(r)

    # 2) Boş kalırsa TOP ile doldur (LONG/SHORT denge)
    remaining = TABLE_SIZE - len(out)
    if remaining > 0:
        long_iter = longs.iterrows()
        short_iter = shorts.iterrows()

        cur_long = sum(1 for x in out if x["YÖN"] == "LONG")
        cur_short = sum(1 for x in out if x["YÖN"] == "SHORT")

        while remaining > 0:
            pick_long = (cur_long <= cur_short)
            picked = False

            if pick_long:
                for _, r in long_iter:
                    if push_row(r):
                        cur_long += 1
                        remaining -= 1
                        picked = True
                        break
                if not picked:
                    for _, r in short_iter:
                        if push_row(r):
                            cur_short += 1
                            remaining -= 1
                            picked = True
                            break
            else:
                for _, r in short_iter:
                    if push_row(r):
                        cur_short += 1
                        remaining -= 1
                        picked = True
                        break
                if not picked:
                    for _, r in long_iter:
                        if push_row(r):
                            cur_long += 1
                            remaining -= 1
                            picked = True
                            break

            if not picked:
                break

    out_df = pd.DataFrame(out)
    if out_df.empty:
        return out_df

    out_df = out_df[["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI"]].copy()

    # tabloda STRONG üstte
    out_df["_STR"] = np.where((out_df["SKOR"] >= STRONG_LONG_MIN) | (out_df["SKOR"] <= STRONG_SHORT_MAX), 1, 0)
    out_df["_SEV"] = np.where(out_df["YÖN"] == "LONG", out_df["SKOR"], 100 - out_df["SKOR"])
    out_df = out_df.sort_values(["_STR", "_SEV", "QV_24H"], ascending=[False, False, False]).drop(columns=["_STR", "_SEV"])
    out_df = out_df.reset_index(drop=True)
    return out_df


# ============================================================
# Styling (Koyu tema, solukluk yok)
# ============================================================
def style_table(df: pd.DataFrame):
    def dir_style(v):
        if v == "LONG":
            return "background-color:#0f3d2e;color:#ffffff;font-weight:800;"
        if v == "SHORT":
            return "background-color:#4a1414;color:#ffffff;font-weight:800;"
        return ""

    def score_style(v):
        v = int(v)
        if v >= STRONG_LONG_MIN:
            return "background-color:#0b5d1e;color:#ffffff;font-weight:900;"
        if v <= STRONG_SHORT_MAX:
            return "background-color:#7a1111;color:#ffffff;font-weight:900;"
        return "background-color:#0b1220;color:#e6edf3;font-weight:800;"

    def raw_style(v):
        v = float(v)
        if v >= 80:
            return "background-color:#083b15;color:#e6edf3;"
        if v <= 20:
            return "background-color:#3b0b0b;color:#e6edf3;"
        return "background-color:#0b1220;color:#e6edf3;"

    fmt = {"FİYAT": "{:.6f}", "RAW": "{:.0f}", "QV_24H": "{:,.0f}", "KAPI": "{:.0f}", "SKOR": "{:.0f}"}

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
            {"selector": "th", "props": [("background-color", "#0b0f14"), ("color", "#e6edf3"), ("border-color", "#1f2a37"), ("font-weight", "900")]},
            {"selector": "td", "props": [("border-color", "#1f2a37")]},
            {"selector": "table", "props": [("border-collapse", "collapse")]},
        ])
    )


# ============================================================
# Streamlit page
# ============================================================
st.set_page_config(page_title="KuCoin PRO Sniper — Auto (LONG+SHORT)", layout="wide")

# Koyu tema (solukluk kilidi)
st.markdown(
    """
<style>
:root { color-scheme: dark !important; }
html, body { background:#0b0f14 !important; }
.stApp { background:#0b0f14 !important; color:#e6edf3 !important; opacity:1 !important; }
.block-container { padding-top: 1.0rem; }
[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
[data-testid="stToolbar"] { display:none; }
h1,h2,h3,h4,h5,h6,p,span,div,label { color:#e6edf3 !important; opacity:1 !important; }
section.main { background:#0b0f14 !important; }
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

def card(text: str, bg: str, border: str):
    st.markdown(
        f"""
<div style="
background:{bg};
border:1px solid {border};
border-radius:14px;
padding:14px 16px;
font-weight:800;
color:#e6edf3;
">
{text}
</div>
""",
        unsafe_allow_html=True,
    )

now_ist = datetime.now(IST_TZ)
st.markdown(
    f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;">
  <div>
    <div style="font-size:30px;font-weight:950;letter-spacing:0.2px;">🎯 KuCoin PRO Sniper — Auto (LONG + SHORT)</div>
    <div style="opacity:0.95;margin-top:6px;">
      TF={TIMEFRAME} • STRONG: SKOR≥{STRONG_LONG_MIN} (LONG) / SKOR≤{STRONG_SHORT_MAX} (SHORT) • 6 Kapı • Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s
    </div>
  </div>
  <div style="text-align:right;opacity:0.95;">
    <div style="font-size:12px;">Istanbul Time</div>
    <div style="font-size:18px;font-weight:900;">{now_ist.strftime('%Y-%m-%d %H:%M:%S')}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# Scan
ex = make_exchange()

with st.spinner("⏳ KuCoin USDT spot evreni taranıyor... (lütfen bekle)"):
    symbols = load_usdt_spot_symbols()

    tickers = safe_fetch_tickers(ex, symbols)
    rows_rank = []
    for s in symbols:
        qv = qv_usdt(tickers.get(s))
        if qv >= MIN_QV_24H_USDT:
            rows_rank.append((s, qv))
    rows_rank.sort(key=lambda x: x[1], reverse=True)

    scan_list = [s for s, _ in rows_rank[:MAX_SCAN]]

    card(f"🧠 Evren (USDT spot): {len(symbols)} • Likidite filtresi sonrası: {len(rows_rank)} • Tarama: {len(scan_list)}", "#0b1a2a", "#1f3a5f")

    prog = st.progress(0, text="Scanning...")
    out_rows = []

    total = max(1, len(scan_list))
    qv_map = dict(rows_rank)

    for i, sym in enumerate(scan_list, start=1):
        try:
            ohlcv = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)

            # ✅ BUG FIX: limit=200 iken 210 kontrolü yüzünden hepsi düşüyordu.
            if not ohlcv or len(ohlcv) < CANDLE_LIMIT:
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])

            raw, gates = compute_raw_and_gates(df)
            last_price = float(df["close"].iloc[-1])
            qv24 = float(qv_map.get(sym, 0.0))

            out_rows.append({
                "COIN": sym.split("/")[0],
                "FİYAT": last_price,
                "RAW": raw,
                "KAPI": gates,
                "QV_24H": qv24,
            })

        except (ccxt.RequestTimeout, ccxt.NetworkError, ccxt.ExchangeError):
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
    card("⚠️ Aday yok (network/KuCoin veya filtre çok sert). Bir sonraki auto refresh’i bekle.", "#2a1b0b", "#5f3a1f")
else:
    table = build_table(df_all)

    strong_count = int(((table["SKOR"] >= STRONG_LONG_MIN) | (table["SKOR"] <= STRONG_SHORT_MAX)).sum())
    if strong_count > 0:
        card("✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu.", "#0f2a19", "#1f5f3a")
    else:
        card("⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu (LONG+SHORT birlikte).", "#2a1b0b", "#5f3a1f")

    st.dataframe(style_table(table), use_container_width=True, height=720)
