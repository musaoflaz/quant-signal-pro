# requirements.txt (install these):
# streamlit==1.42.0
# pandas==2.2.3
# numpy==2.1.3
# ccxt==4.4.97

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import ccxt


# =========================================================
# BASE (DO NOT BREAK) — KuCoin PRO Sniper style
# =========================================================
IST = ZoneInfo("Europe/Istanbul")

TF = "15m"
HTF = "1h"
AUTO_REFRESH_SEC = 240  # 4 dk
TOP_N = 150             # her borsadan ilk 150 USDT
SHOW_ROWS = 20          # tablo sabit 20 satır
BALANCED_FILL = True    # 10 LONG + 10 SHORT hedefi (uygunsa)

STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10
SCORE_STEP = 5          # skor 5'lik adımda

CANDLE_LIMIT_TF = 220
CANDLE_LIMIT_HTF = 180

# RAW 0-100 ağırlıkları
W_TREND = 15
W_RSI   = 25
W_BB    = 20
W_ADX   = 15
W_HTF   = 15
W_LIQ   = 10

# Likidite eşiği (kucoin/okx için pratik)
MIN_QV_WEAK = 250_000
MIN_QV_OK   = 1_000_000
MIN_QV_STR  = 5_000_000


# =========================================================
# PURE NumPy/Pandas INDICATORS
# =========================================================
def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    avg_gain = _ema(gain, period)
    avg_loss = _ema(loss, period) + 1e-12
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def sma(close: np.ndarray, period: int = 20) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    out = np.full_like(close, np.nan, dtype=float)
    if len(close) < period:
        return out
    c = np.cumsum(close, dtype=float)
    c[period:] = c[period:] - c[:-period]
    out[period - 1 :] = c[period - 1 :] / period
    return out

def bollinger(close: np.ndarray, period: int = 20, mult: float = 2.0):
    close = np.asarray(close, dtype=float)
    mid = sma(close, period)
    std = np.full_like(close, np.nan, dtype=float)
    if len(close) >= period:
        for i in range(period - 1, len(close)):
            w = close[i - period + 1 : i + 1]
            std[i] = np.std(w, ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    return mid, upper, lower

def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ]
    )
    tr[0] = high[0] - low[0]

    atr = _ema(tr, period) + 1e-12
    plus_di = 100.0 * (_ema(plus_dm, period) / atr)
    minus_di = 100.0 * (_ema(minus_dm, period) / atr)

    dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12))
    return _ema(dx, period)


# =========================================================
# EXCHANGE HELPERS (CCXT)
# =========================================================
def make_exchange(exchange_id: str) -> ccxt.Exchange:
    klass = getattr(ccxt, exchange_id)
    return klass({"enableRateLimit": True, "timeout": 20000})

def safe_load_markets(ex: ccxt.Exchange) -> dict:
    try:
        return ex.load_markets()
    except Exception:
        return {}

def safe_fetch_tickers(ex: ccxt.Exchange) -> dict:
    try:
        return ex.fetch_tickers()
    except Exception:
        return {}

def is_spot_usdt_market(m: dict) -> bool:
    if not m:
        return False
    if not m.get("active", True):
        return False
    if not m.get("spot", False):
        return False
    return m.get("quote") == "USDT"

def float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None

def ticker_quote_volume(t: dict) -> float:
    if not isinstance(t, dict):
        return 0.0
    qv = float_or_none(t.get("quoteVolume"))
    if qv is not None:
        return qv
    bv = float_or_none(t.get("baseVolume"))
    last = float_or_none(t.get("last"))
    if bv is not None and last is not None:
        return bv * last
    return 0.0

def pick_top_symbols(ex_id: str, top_n: int) -> tuple[list[str], dict, str | None]:
    """
    returns: (symbols, tickers, error_text)
    """
    try:
        ex = make_exchange(ex_id)
        markets = safe_load_markets(ex)
        tickers = safe_fetch_tickers(ex)

        syms = []
        for sym, m in markets.items():
            if is_spot_usdt_market(m) and sym.endswith("/USDT"):
                syms.append(sym)

        ranked = []
        for s in syms:
            ranked.append((s, ticker_quote_volume(tickers.get(s, {}))))
        ranked.sort(key=lambda x: x[1], reverse=True)
        out = [s for s, _ in ranked[:top_n]]
        return out, tickers, None
    except Exception as e:
        return [], {}, f"{type(e).__name__}: {e}"

def fetch_ohlcv_df(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 60:
            return None
        return pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    except Exception:
        return None


# =========================================================
# SCORE (RAW 0-100) + 6 KAPI + 5'lik SKOR
# FIXES:
#  - STRONG artık KAPI==6 olmadan True OLAMAZ
#  - Yön seçimi LONG'a kaymasın diye LONG ve SHORT RAW ayrı hesaplanır, en güçlü olan seçilir
# =========================================================
def round_step(x: int, step: int) -> int:
    return int(np.clip(int(round(x / step) * step), 0, 100))

def _liq_points(qv24: float) -> int:
    liq_pts = int(W_LIQ * 0.25)
    if qv24 >= MIN_QV_STR:
        liq_pts = W_LIQ
    elif qv24 >= MIN_QV_OK:
        liq_pts = int(W_LIQ * 0.7)
    elif qv24 >= MIN_QV_WEAK:
        liq_pts = int(W_LIQ * 0.45)
    return liq_pts

def _score_for_direction(direction: str, df_tf: pd.DataFrame, df_htf: pd.DataFrame | None, qv24: float) -> dict:
    c = df_tf["close"].to_numpy(dtype=float)
    h = df_tf["high"].to_numpy(dtype=float)
    l = df_tf["low"].to_numpy(dtype=float)

    last = float(c[-1])

    r = rsi(c, 14)
    s20 = sma(c, 20)
    _, bbu, bbl = bollinger(c, 20, 2.0)
    a = adx(h, l, c, 14)

    last_rsi = float(r[-1])
    last_sma = float(s20[-1]) if np.isfinite(s20[-1]) else np.nan
    last_bbu = float(bbu[-1]) if np.isfinite(bbu[-1]) else np.nan
    last_bbl = float(bbl[-1]) if np.isfinite(bbl[-1]) else np.nan
    last_adx = float(a[-1])

    liq_pts = _liq_points(qv24)

    raw = 50

    # Trend (SMA20): LONG için sma altı "değer" / SHORT için sma üstü "değer"
    if np.isfinite(last_sma):
        if direction == "LONG":
            raw += W_TREND if last < last_sma else -W_TREND
        else:
            raw += W_TREND if last > last_sma else -W_TREND

    # RSI
    if direction == "LONG":
        if last_rsi <= 35:
            raw += W_RSI
        elif last_rsi <= 45:
            raw += int(W_RSI * 0.6)
        elif last_rsi >= 65:
            raw -= W_RSI
        else:
            raw += int(W_RSI * 0.35)
    else:
        if last_rsi >= 65:
            raw += W_RSI
        elif last_rsi >= 55:
            raw += int(W_RSI * 0.6)
        elif last_rsi <= 35:
            raw -= W_RSI
        else:
            raw += int(W_RSI * 0.35)

    # Bollinger
    if direction == "LONG":
        if np.isfinite(last_bbl) and last <= last_bbl * 1.01:
            raw += W_BB
        elif np.isfinite(last_bbu) and last >= last_bbu * 0.99:
            raw -= W_BB
        else:
            raw += int(W_BB * 0.35)
    else:
        if np.isfinite(last_bbu) and last >= last_bbu * 0.99:
            raw += W_BB
        elif np.isfinite(last_bbl) and last <= last_bbl * 1.01:
            raw -= W_BB
        else:
            raw += int(W_BB * 0.35)

    # ADX (trend strength)
    if last_adx >= 25:
        raw += W_ADX
    elif last_adx >= 20:
        raw += int(W_ADX * 0.7)
    else:
        raw += int(W_ADX * 0.2)

    # HTF confirmation (1h)
    htf_ok = False
    if df_htf is not None and len(df_htf) >= 60:
        c1 = df_htf["close"].to_numpy(dtype=float)
        r1 = rsi(c1, 14)
        s1 = sma(c1, 20)
        if np.isfinite(s1[-1]):
            if direction == "LONG":
                if float(r1[-1]) <= 60 and float(c1[-1]) <= float(s1[-1]) * 1.02:
                    htf_ok = True
            else:
                if float(r1[-1]) >= 40 and float(c1[-1]) >= float(s1[-1]) * 0.98:
                    htf_ok = True
    raw += W_HTF if htf_ok else int(W_HTF * 0.25)

    # Liquidity to raw
    raw += liq_pts

    raw = int(np.clip(raw, 0, 100))
    score = round_step(raw, SCORE_STEP)

    # 6 KAPI (direction-aware on HTF + basic)
    gates = 0
    gates += 1 if (last_rsi <= 45 or last_rsi >= 55) else 0
    gates += 1 if ((np.isfinite(last_bbl) and last <= last_bbl * 1.01) or (np.isfinite(last_bbu) and last >= last_bbu * 0.99)) else 0
    gates += 1 if (np.isfinite(last_sma) and (last < last_sma or last > last_sma * 1.01)) else 0
    gates += 1 if (last_adx >= 20) else 0
    gates += 1 if htf_ok else 0
    gates += 1 if (qv24 >= MIN_QV_WEAK) else 0

    # FIX: STRONG = gate==6 + eşik
    strong = False
    if int(gates) == 6:
        if direction == "LONG" and score >= STRONG_LONG_MIN:
            strong = True
        if direction == "SHORT" and score <= STRONG_SHORT_MAX:
            strong = True

    return {
        "direction": direction,
        "raw": raw,
        "score": int(score),
        "price": float(last),
        "qv24": float(qv24),
        "gates": int(gates),
        "strong": bool(strong),
    }

def compute_best_pack(df_tf: pd.DataFrame, df_htf: pd.DataFrame | None, qv24: float) -> dict:
    long_pack = _score_for_direction("LONG", df_tf, df_htf, qv24)
    short_pack = _score_for_direction("SHORT", df_tf, df_htf, qv24)

    # direction choice by "conviction"
    # - LONG conviction = score (higher better)
    # - SHORT conviction = (100 - score) (lower better)
    long_conv = long_pack["score"]
    short_conv = 100 - short_pack["score"]

    # If equal, prefer the one with higher gates (rare)
    if long_conv > short_conv:
        return long_pack
    if short_conv > long_conv:
        return short_pack

    return long_pack if long_pack["gates"] >= short_pack["gates"] else short_pack


# =========================================================
# BUILD ROWS (KuCoin + OKX) + BOTH confirmation
# FIX:
#  - BOTH sadece iki borsada da STRONG + KAPI==6 + aynı yön ise
# =========================================================
def build_rows() -> tuple[pd.DataFrame, dict]:
    ku_syms, ku_tickers, ku_err = pick_top_symbols("kucoin", TOP_N)
    ok_syms, ok_tickers, ok_err = pick_top_symbols("okx", TOP_N)

    status = {
        "kucoin_ok": ku_err is None and len(ku_syms) > 0,
        "okx_ok": ok_err is None and len(ok_syms) > 0,
        "ku_err": ku_err,
        "ok_err": ok_err,
    }

    ex_ku = make_exchange("kucoin") if status["kucoin_ok"] else None
    ex_ok = make_exchange("okx") if status["okx_ok"] else None
    if ex_ku:
        safe_load_markets(ex_ku)
    if ex_ok:
        safe_load_markets(ex_ok)

    ku_set = set(ku_syms)
    ok_set = set(ok_syms)
    all_syms = sorted(ku_set.union(ok_set))

    rows = []
    prog = st.progress(0.0)
    info = st.empty()

    total = max(1, len(all_syms))
    for i, sym in enumerate(all_syms, start=1):
        prog.progress(min(i / total, 1.0))
        if i == 1 or i % 10 == 0 or i == total:
            info.info(f"Taranıyor: {i}/{total} • {sym}")

        ku_pack = None
        ok_pack = None

        if ex_ku and sym in ku_set:
            qv = ticker_quote_volume(ku_tickers.get(sym, {}))
            df15 = fetch_ohlcv_df(ex_ku, sym, TF, CANDLE_LIMIT_TF)
            if df15 is not None:
                df1h = fetch_ohlcv_df(ex_ku, sym, HTF, CANDLE_LIMIT_HTF)
                ku_pack = compute_best_pack(df15, df1h, qv)

        if ex_ok and sym in ok_set:
            qv = ticker_quote_volume(ok_tickers.get(sym, {}))
            df15 = fetch_ohlcv_df(ex_ok, sym, TF, CANDLE_LIMIT_TF)
            if df15 is not None:
                df1h = fetch_ohlcv_df(ex_ok, sym, HTF, CANDLE_LIMIT_HTF)
                ok_pack = compute_best_pack(df15, df1h, qv)

        # BOTH (only if both STRONG + same direction)
        if ku_pack and ok_pack and ku_pack["strong"] and ok_pack["strong"] and ku_pack["direction"] == ok_pack["direction"]:
            direction = ku_pack["direction"]

            # conservative: use weaker conviction among the two
            if direction == "LONG":
                raw = int(min(ku_pack["raw"], ok_pack["raw"]))
            else:
                raw = int(max(ku_pack["raw"], ok_pack["raw"]))

            score = round_step(raw, SCORE_STEP)
            pick = ku_pack if ku_pack["qv24"] >= ok_pack["qv24"] else ok_pack
            price = pick["price"]
            qv24 = pick["qv24"]
            gates = int(min(ku_pack["gates"], ok_pack["gates"]))  # should be 6 anyway

            # enforce gates==6 for BOTH too
            both_strong = (gates == 6) and (
                (direction == "LONG" and score >= STRONG_LONG_MIN) or (direction == "SHORT" and score <= STRONG_SHORT_MAX)
            )

            rows.append(
                {
                    "YÖN": direction,
                    "COIN": sym.replace("/USDT", ""),
                    "SKOR": int(score),
                    "FİYAT": float(price),
                    "RAW": int(raw),
                    "QV_24H": float(qv24),
                    "KAPI": int(gates),
                    "STRONG": bool(both_strong),
                    "SOURCE": "BOTH",
                }
            )
        else:
            if ku_pack:
                rows.append(
                    {
                        "YÖN": ku_pack["direction"],
                        "COIN": sym.replace("/USDT", ""),
                        "SKOR": int(ku_pack["score"]),
                        "FİYAT": float(ku_pack["price"]),
                        "RAW": int(ku_pack["raw"]),
                        "QV_24H": float(ku_pack["qv24"]),
                        "KAPI": int(ku_pack["gates"]),
                        "STRONG": bool(ku_pack["strong"]),
                        "SOURCE": "KUCOIN",
                    }
                )
            if ok_pack:
                rows.append(
                    {
                        "YÖN": ok_pack["direction"],
                        "COIN": sym.replace("/USDT", ""),
                        "SKOR": int(ok_pack["score"]),
                        "FİYAT": float(ok_pack["price"]),
                        "RAW": int(ok_pack["raw"]),
                        "QV_24H": float(ok_pack["qv24"]),
                        "KAPI": int(ok_pack["gates"]),
                        "STRONG": bool(ok_pack["strong"]),
                        "SOURCE": "OKX",
                    }
                )

        time.sleep(0.01)

    prog.empty()
    info.empty()

    df = pd.DataFrame(rows)
    return df, status


# =========================================================
# TABLE PICKING (STRONG first, then TOP fill to 20)
# FIX:
#  - SHORT'ların gelmesi için direction seçimi iyileştirildi (üstte)
#  - Doldurma mantığı aynen, 10/10 denge korunur
# =========================================================
def pick_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    # "yakınlık" metriği: LONG için RAW büyük, SHORT için RAW küçük daha iyi
    df["sev"] = np.where(df["YÖN"] == "LONG", df["RAW"], 100 - df["RAW"])

    strong = df[df["STRONG"] == True].copy()
    non = df[df["STRONG"] == False].copy()

    strong_long = strong[strong["YÖN"] == "LONG"].sort_values(["RAW", "QV_24H"], ascending=[False, False])
    strong_short = strong[strong["YÖN"] == "SHORT"].sort_values(["RAW", "QV_24H"], ascending=[True, False])

    picked = pd.concat([strong_long, strong_short], axis=0)
    if len(picked) > SHOW_ROWS:
        picked = picked.head(SHOW_ROWS)

    remaining_slots = SHOW_ROWS - len(picked)
    if remaining_slots <= 0:
        return picked.drop(columns=["sev"], errors="ignore").reset_index(drop=True)

    non_long = non[non["YÖN"] == "LONG"].sort_values(["RAW", "QV_24H"], ascending=[False, False])
    non_short = non[non["YÖN"] == "SHORT"].sort_values(["RAW", "QV_24H"], ascending=[True, False])

    fill = pd.DataFrame()

    if BALANCED_FILL:
        need_long = 10 - int((picked["YÖN"] == "LONG").sum())
        need_short = 10 - int((picked["YÖN"] == "SHORT").sum())
        need_long = max(0, need_long)
        need_short = max(0, need_short)

        fill_long = non_long.head(need_long)
        fill_short = non_short.head(need_short)
        fill = pd.concat([fill_long, fill_short], axis=0)

        remaining_slots = SHOW_ROWS - (len(picked) + len(fill))
        if remaining_slots > 0:
            rest = pd.concat([non_long.iloc[need_long:], non_short.iloc[need_short:]], axis=0)
            rest = rest.sort_values(["sev", "QV_24H"], ascending=[False, False]).head(remaining_slots)
            fill = pd.concat([fill, rest], axis=0)
    else:
        fill = non.sort_values(["sev", "QV_24H"], ascending=[False, False]).head(remaining_slots)

    out = pd.concat([picked, fill], axis=0).head(SHOW_ROWS)
    return out.drop(columns=["sev"], errors="ignore").reset_index(drop=True)


# =========================================================
# STYLING (dark theme, long green, short red, strong darker)
# =========================================================
def style_table(df: pd.DataFrame):
    def row_style(row):
        direction = row.get("YÖN", "")
        is_strong = bool(row.get("STRONG", False))

        if direction == "LONG":
            return ["background-color: #064e3b; color: #e6edf3; font-weight: 700;" if is_strong
                    else "background-color: rgba(6,78,59,0.55); color: #e6edf3; font-weight: 700;"] * len(row)
        if direction == "SHORT":
            return ["background-color: #7f1d1d; color: #e6edf3; font-weight: 700;" if is_strong
                    else "background-color: rgba(127,29,29,0.55); color: #e6edf3; font-weight: 700;"] * len(row)
        return [""] * len(row)

    fmt = {
        "FİYAT": "{:.6f}",
        "QV_24H": "{:,.0f}",
        "RAW": "{:.0f}",
        "SKOR": "{:.0f}",
        "KAPI": "{:.0f}",
    }

    return (
        df.style
        .format(fmt)
        .apply(row_style, axis=1)
        .set_properties(**{"border-color": "#1f2a37"})
    )


# =========================================================
# STREAMLIT APP (AUTO)
# =========================================================
st.set_page_config(page_title="KuCoin PRO Sniper — Auto (KuCoin + OKX)", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.2rem; }
h1, h2, h3, h4, h5, h6, p, span, div { color: #e6edf3; }
[data-testid="stSidebar"] { display: none; }  /* no manual menu */
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎯 KuCoin PRO Sniper — Auto (KuCoin + OKX)")

st.caption(
    f"TF={TF} • HTF={HTF} • STRONG: SKOR≥{STRONG_LONG_MIN} (LONG) / SKOR≤{STRONG_SHORT_MAX} (SHORT) • "
    f"6 Kapı şart • Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s • Istanbul Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}"
)

with st.spinner("🧠 KuCoin + OKX taranıyor..."):
    df_all, status = build_rows()

c1, c2 = st.columns(2)
with c1:
    if status["kucoin_ok"]:
        st.success("KuCoin: ✅ Bağlandı")
    else:
        st.error(f"KuCoin: ❌ Hata — {status['ku_err']}")
with c2:
    if status["okx_ok"]:
        st.success("OKX: ✅ Bağlandı")
    else:
        st.error(f"OKX: ❌ Hata — {status['ok_err']}")

if df_all.empty:
    st.warning("Tablo boş geldi (network/limit). Bir sonraki auto refresh’i bekle.")
    time.sleep(AUTO_REFRESH_SEC)
    st.rerun()

df_show = pick_display(df_all)

# Counters
strong_long = int(((df_show["YÖN"] == "LONG") & (df_show["STRONG"] == True)).sum())
strong_short = int(((df_show["YÖN"] == "SHORT") & (df_show["STRONG"] == True)).sum())
longs = int((df_show["YÖN"] == "LONG").sum())
shorts = int((df_show["YÖN"] == "SHORT").sum())

st.info(
    f"✅ STRONG LONG: {strong_long}  |  💀 STRONG SHORT: {strong_short}  |  LONG: {longs}  |  SHORT: {shorts}"
)

# Banner
if int(df_all["STRONG"].sum()) > 0:
    st.success("✅ STRONG bulundu (KAPI=6). Kalan boşluklar TOP adaylarla dolduruldu.")
else:
    st.warning("⚠️ Şu an STRONG yok (KAPI=6 + eşik). Yine de TOP adaylarla tablo dolu.")

# Table
cols = ["YÖN","COIN","SKOR","FİYAT","RAW","QV_24H","KAPI","STRONG","SOURCE"]
df_show = df_show.loc[:, cols].copy()

st.dataframe(style_table(df_show), use_container_width=True, height=650)

# Auto refresh
time.sleep(AUTO_REFRESH_SEC)
st.rerun()
