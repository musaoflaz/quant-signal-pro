# app.py
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import ccxt


# =========================
# CONFIG
# =========================
APP_TITLE = "Dual Exchange Sniper — KuCoin + Binance (Spot USDT)"
TF = "15m"
HTF = "1h"
LIMIT_TF = 220
LIMIT_HTF = 180

TOP_N = 150
REFRESH_SEC = 240  # 4 dk

STRONG_LONG = 90
STRONG_SHORT = 10

# Score weights (tweak later without breaking UI)
W_RSI = 25
W_BB = 20
W_SMA = 15
W_ADX = 15
W_HTF = 15
W_LIQ = 10  # volume/spread gate


# =========================
# INDICATORS (vectorized)
# =========================
def _ema(x: np.ndarray, period: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
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


def sma(x: np.ndarray, period: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < period:
        return np.full_like(x, np.nan)
    c = np.cumsum(x, dtype=float)
    c[period:] = c[period:] - c[:-period]
    out = np.full_like(x, np.nan, dtype=float)
    out[period - 1 :] = c[period - 1 :] / period
    return out


def bollinger(close: np.ndarray, period: int = 20, mult: float = 2.0):
    close = np.asarray(close, dtype=float)
    ma = sma(close, period)
    # rolling std via convolution-ish trick (safe & simple)
    out_std = np.full_like(close, np.nan, dtype=float)
    if len(close) >= period:
        for i in range(period - 1, len(close)):
            w = close[i - period + 1 : i + 1]
            out_std[i] = np.std(w, ddof=0)
    upper = ma + mult * out_std
    lower = ma - mult * out_std
    return ma, upper, lower


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


# =========================
# SCORING
# =========================
def score_symbol(df15: pd.DataFrame, df1h: pd.DataFrame | None, vol_quote_24h: float | None) -> dict:
    """
    Returns dict: score, raw, gates, signal, is_strong, details
    """
    c = df15["close"].to_numpy(dtype=float)
    h = df15["high"].to_numpy(dtype=float)
    l = df15["low"].to_numpy(dtype=float)

    r = rsi(c, 14)
    ma20, bb_u, bb_l = bollinger(c, 20, 2.0)
    s20 = sma(c, 20)
    a = adx(h, l, c, 14)

    last = float(c[-1])
    last_rsi = float(r[-1])
    last_sma = float(s20[-1]) if np.isfinite(s20[-1]) else np.nan
    last_bbu = float(bb_u[-1]) if np.isfinite(bb_u[-1]) else np.nan
    last_bbl = float(bb_l[-1]) if np.isfinite(bb_l[-1]) else np.nan
    last_adx = float(a[-1])

    # Direction bias (simple)
    # LONG if oversold / below bands / below sma; SHORT if opposite
    long_bias = 0
    short_bias = 0

    # RSI contribution
    rsi_pts = 0
    if last_rsi <= 30:
        long_bias += 1
        rsi_pts = W_RSI
    elif last_rsi <= 40:
        long_bias += 1
        rsi_pts = int(W_RSI * 0.6)
    elif last_rsi >= 70:
        short_bias += 1
        rsi_pts = 0
    elif last_rsi >= 60:
        short_bias += 1
        rsi_pts = int(W_RSI * 0.2)
    else:
        rsi_pts = int(W_RSI * 0.35)

    # BB contribution
    bb_pts = int(W_BB * 0.35)
    if np.isfinite(last_bbl) and last <= last_bbl * 1.01:
        long_bias += 1
        bb_pts = W_BB
    if np.isfinite(last_bbu) and last >= last_bbu * 0.99:
        short_bias += 1
        bb_pts = 0

    # SMA contribution
    sma_pts = int(W_SMA * 0.35)
    if np.isfinite(last_sma):
        if last < last_sma:
            long_bias += 1
            sma_pts = W_SMA
        elif last > last_sma * 1.01:
            short_bias += 1
            sma_pts = 0

    # ADX contribution (trend strength)
    adx_pts = int(W_ADX * 0.35)
    if last_adx >= 25:
        adx_pts = W_ADX
    elif last_adx >= 20:
        adx_pts = int(W_ADX * 0.7)

    # HTF confirmation
    htf_pts = int(W_HTF * 0.35)
    htf_ok = False
    if df1h is not None and len(df1h) >= 60:
        c1 = df1h["close"].to_numpy(dtype=float)
        r1 = rsi(c1, 14)
        s1 = sma(c1, 20)
        if np.isfinite(s1[-1]):
            # confirm direction: if we want LONG, prefer HTF RSI not overbought and price <= SMA20-ish
            # if SHORT, prefer HTF RSI not oversold and price >= SMA20-ish
            if long_bias >= short_bias:
                if float(r1[-1]) <= 60 and float(c1[-1]) <= float(s1[-1]) * 1.02:
                    htf_ok = True
            else:
                if float(r1[-1]) >= 40 and float(c1[-1]) >= float(s1[-1]) * 0.98:
                    htf_ok = True
        if htf_ok:
            htf_pts = W_HTF

    # Liquidity (simple: quote volume threshold)
    liq_pts = int(W_LIQ * 0.25)
    if vol_quote_24h is not None:
        # tune thresholds if needed
        if vol_quote_24h >= 5_000_000:
            liq_pts = W_LIQ
        elif vol_quote_24h >= 1_000_000:
            liq_pts = int(W_LIQ * 0.7)
        elif vol_quote_24h >= 250_000:
            liq_pts = int(W_LIQ * 0.45)

    total = rsi_pts + bb_pts + sma_pts + adx_pts + htf_pts + liq_pts
    total = int(np.clip(total, 0, 100))

    # Decide signal
    if long_bias > short_bias:
        signal = "LONG"
    elif short_bias > long_bias:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    # Gates count (visual “KAPI”)
    gates = 0
    gates += 1 if last_rsi <= 40 or last_rsi >= 60 else 0
    gates += 1 if (np.isfinite(last_bbl) and last <= last_bbl * 1.01) or (np.isfinite(last_bbu) and last >= last_bbu * 0.99) else 0
    gates += 1 if np.isfinite(last_sma) and (last < last_sma or last > last_sma * 1.01) else 0
    gates += 1 if last_adx >= 20 else 0
    gates += 1 if htf_ok else 0
    gates += 1 if (vol_quote_24h is not None and vol_quote_24h >= 250_000) else 0

    is_strong = (total >= STRONG_LONG) or (total <= STRONG_SHORT)

    return {
        "score": total,
        "signal": signal,
        "gates": gates,
        "is_strong": is_strong,
        "last_rsi": round(last_rsi, 2),
        "last_adx": round(last_adx, 2),
        "htf_ok": htf_ok,
    }


# =========================
# EXCHANGE HELPERS
# =========================
def make_exchange(exchange_id: str):
    klass = getattr(ccxt, exchange_id)
    ex = klass({"enableRateLimit": True})
    ex.load_markets()
    return ex


def safe_fetch_tickers(ex):
    try:
        return ex.fetch_tickers()
    except Exception:
        # fallback: some exchanges can be picky
        return {}


def normalize_symbol(sym: str) -> str:
    # ccxt spot symbols are usually "ABC/USDT"
    return sym.strip()


def top_usdt_symbols_from_tickers(tickers: dict, n: int) -> list[str]:
    rows = []
    for sym, t in tickers.items():
        sym = normalize_symbol(sym)
        if not sym.endswith("/USDT"):
            continue
        if t is None:
            continue
        qv = t.get("quoteVolume", None)
        bv = t.get("baseVolume", None)
        last = t.get("last", None)

        # robust float conversions
        def f(x):
            try:
                return float(x)
            except Exception:
                return None

        qv = f(qv)
        bv = f(bv)
        last = f(last)

        if qv is None:
            if bv is not None and last is not None:
                qv = bv * last
            else:
                qv = 0.0

        rows.append((sym, qv))

    rows.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in rows[:n]]


def fetch_ohlcv_df(ex, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 50:
            return None
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        return df
    except Exception:
        return None


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

colA, colB, colC, colD = st.columns([1.2, 1, 1, 1])
with colA:
    auto = st.toggle("Auto Refresh", value=True)
with colB:
    top_n = st.number_input("Top N (USDT)", min_value=30, max_value=300, value=TOP_N, step=10)
with colC:
    tf = st.selectbox("TF", ["5m", "15m", "30m", "1h"], index=1)
with colD:
    refresh = st.number_input("Refresh (sec)", min_value=60, max_value=900, value=REFRESH_SEC, step=30)

st.caption(f"TF={tf} • HTF={HTF} • STRONG: Score≥{STRONG_LONG} (LONG) / Score≤{STRONG_SHORT} (SHORT) • Istanbul Time: {datetime.now(timezone(timedelta(hours=3))).strftime('%Y-%m-%d %H:%M:%S')}")

status_box = st.empty()
progress = st.progress(0)
logline = st.empty()


@st.cache_data(ttl=120, show_spinner=False)
def connect_and_get_tickers():
    out = {}
    errors = {}

    # Binance
    try:
        bx = make_exchange("binance")
        bt = safe_fetch_tickers(bx)
        out["binance"] = {"ex": None, "tickers": bt}
        errors["binance"] = None
    except Exception as e:
        out["binance"] = {"ex": None, "tickers": {}}
        errors["binance"] = str(e)

    # KuCoin
    try:
        kx = make_exchange("kucoin")
        kt = safe_fetch_tickers(kx)
        out["kucoin"] = {"ex": None, "tickers": kt}
        errors["kucoin"] = None
    except Exception as e:
        out["kucoin"] = {"ex": None, "tickers": {}}
        errors["kucoin"] = str(e)

    return out, errors


def banner_status(errors: dict):
    ok_bin = errors.get("binance") is None
    ok_kuc = errors.get("kucoin") is None

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.markdown(f"**Binance:** {'✅ Bağlandı' if ok_bin else '❌ Hata'}")
        if not ok_bin:
            st.caption(errors.get("binance"))
    with c2:
        st.markdown(f"**KuCoin:** {'✅ Bağlandı' if ok_kuc else '❌ Hata'}")
        if not ok_kuc:
            st.caption(errors.get("kucoin"))
    with c3:
        if ok_bin and ok_kuc:
            st.success("İki borsa da bağlı. Tabloda SOURCE = Both / Binance / KuCoin göreceksin.")
        elif ok_bin or ok_kuc:
            st.warning("Tek borsa bağlı. SOURCE kolonundan hangisinin çalıştığı net görünür.")
        else:
            st.error("İki borsa da bağlanamadı. Logs kontrol et.")


def run_scan(top_n: int, tf: str):
    # connect tickers (cached)
    pack, errors = connect_and_get_tickers()

    status_box.container()
    with status_box.container():
        banner_status(errors)

    # If both dead, stop
    if errors.get("binance") is not None and errors.get("kucoin") is not None:
        return pd.DataFrame()

    # re-create live exchange objects for OHLCV (NOT cached)
    ex_bin = None
    ex_kuc = None
    if errors.get("binance") is None:
        ex_bin = make_exchange("binance")
    if errors.get("kucoin") is None:
        ex_kuc = make_exchange("kucoin")

    btickers = pack["binance"]["tickers"]
    ktickers = pack["kucoin"]["tickers"]

    bin_syms = top_usdt_symbols_from_tickers(btickers, top_n) if ex_bin else []
    kuc_syms = top_usdt_symbols_from_tickers(ktickers, top_n) if ex_kuc else []

    set_bin = set(bin_syms)
    set_kuc = set(kuc_syms)

    all_syms = sorted(set_bin.union(set_kuc))
    total_jobs = len(all_syms)
    if total_jobs == 0:
        return pd.DataFrame()

    rows = []
    done = 0

    for sym in all_syms:
        done += 1
        progress.progress(min(done / total_jobs, 1.0))
        if done % 10 == 0 or done == 1:
            logline.info(f"Taranıyor: {done}/{total_jobs} • {sym}")

        in_bin = sym in set_bin and ex_bin is not None
        in_kuc = sym in set_kuc and ex_kuc is not None

        b = {}
        k = {}

        # KuCoin side
        if in_kuc:
            t = ktickers.get(sym, {}) or {}
            qv = t.get("quoteVolume", None)
            try:
                qv = float(qv) if qv is not None else None
            except Exception:
                qv = None

            df15 = fetch_ohlcv_df(ex_kuc, sym, tf, LIMIT_TF)
            df1h = None
            if df15 is not None:
                df1h = fetch_ohlcv_df(ex_kuc, sym, HTF, LIMIT_HTF)
                k = score_symbol(df15, df1h, qv)
                k["price"] = float(df15["close"].iloc[-1])
                k["qv24h"] = qv

        # Binance side
        if in_bin:
            t = btickers.get(sym, {}) or {}
            qv = t.get("quoteVolume", None)
            try:
                qv = float(qv) if qv is not None else None
            except Exception:
                qv = None

            df15 = fetch_ohlcv_df(ex_bin, sym, tf, LIMIT_TF)
            df1h = None
            if df15 is not None:
                df1h = fetch_ohlcv_df(ex_bin, sym, HTF, LIMIT_HTF)
                b = score_symbol(df15, df1h, qv)
                b["price"] = float(df15["close"].iloc[-1])
                b["qv24h"] = qv

        # Decide SOURCE
        if in_bin and in_kuc:
            source = "Both"
        elif in_bin:
            source = "Binance"
        else:
            source = "KuCoin"

        # Merge scores
        k_score = k.get("score", None)
        b_score = b.get("score", None)

        # Final score:
        # - If both: take min(score) for "confirmed strength" (more conservative)
        # - else: take the available one
        if source == "Both" and (k_score is not None) and (b_score is not None):
            final_score = int(min(k_score, b_score))
        else:
            final_score = int(k_score if k_score is not None else (b_score if b_score is not None else 0))

        # Final signal (prefer both consistent)
        k_sig = k.get("signal", None)
        b_sig = b.get("signal", None)
        if source == "Both" and k_sig and b_sig:
            final_sig = k_sig if k_sig == b_sig else "MIXED"
        else:
            final_sig = k_sig if k_sig else (b_sig if b_sig else "N/A")

        # Strong logic:
        # - Dual confirmed strong if BOTH have strong condition
        dual_confirmed = False
        if source == "Both" and (k_score is not None) and (b_score is not None):
            dual_confirmed = (k_score >= STRONG_LONG and b_score >= STRONG_LONG) or (k_score <= STRONG_SHORT and b_score <= STRONG_SHORT)

        # Any-strong
        any_strong = False
        if k_score is not None:
            any_strong |= (k_score >= STRONG_LONG) or (k_score <= STRONG_SHORT)
        if b_score is not None:
            any_strong |= (b_score >= STRONG_LONG) or (b_score <= STRONG_SHORT)

        # pick a display price/volume
        price = k.get("price", None) if k.get("price", None) is not None else b.get("price", None)
        qv = k.get("qv24h", None) if k.get("qv24h", None) is not None else b.get("qv24h", None)

        rows.append(
            {
                "SYMBOL": sym,
                "SOURCE": source,
                "KUCOIN_SCORE": k_score,
                "BINANCE_SCORE": b_score,
                "FINAL_SCORE": final_score,
                "SIGNAL": final_sig,
                "PRICE": price,
                "QV_24H": qv,
                "DUAL_CONFIRMED": dual_confirmed,
                "STRONG": any_strong,
                "KAPI_KUCOIN": k.get("gates", None),
                "KAPI_BINANCE": b.get("gates", None),
            }
        )

    df = pd.DataFrame(rows)

    # Rank: show strongest first
    df["RANK_KEY"] = df["FINAL_SCORE"].fillna(0).astype(int)
    df = df.sort_values(["STRONG", "DUAL_CONFIRMED", "RANK_KEY"], ascending=[False, False, False]).drop(columns=["RANK_KEY"])

    return df


# =========================
# RUN
# =========================
with st.spinner("Taranıyor..."):
    df = run_scan(int(top_n), tf)

progress.empty()
logline.empty()

if df is None or df.empty:
    st.warning("Tablo boş geldi. (Bağlantı sorunu veya veri çekilemedi)")
    st.stop()

# Counters
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Toplam Satır", len(df))
with c2:
    st.metric("STRONG (any)", int(df["STRONG"].sum()))
with c3:
    st.metric("DUAL CONFIRMED", int(df["DUAL_CONFIRMED"].sum()))
with c4:
    st.metric("Both", int((df["SOURCE"] == "Both").sum()))

# Info bar like your "final" style
if int(df["DUAL_CONFIRMED"].sum()) > 0:
    st.success("✅ DUAL CONFIRMED bulundu. Tabloda koyu yeşil satırlar öncelikli.")
elif int(df["STRONG"].sum()) > 0:
    st.info("✅ STRONG bulundu. SOURCE kolonundan Binance/KuCoin/Both net görünür.")
else:
    st.warning("⚠️ Şu an STRONG yok. Yine de tablo dolu — fırsatlar takipte.")

# Styling
def style_rows(row):
    # dark green if dual confirmed
    if row.get("DUAL_CONFIRMED", False):
        return ["background-color: rgba(0, 90, 0, 0.35);"] * len(row)
    # bright green if strong long single exchange
    fs = row.get("FINAL_SCORE", 0)
    if fs >= STRONG_LONG and row.get("SOURCE") != "Both":
        return ["background-color: rgba(0, 150, 0, 0.25);"] * len(row)
    # red if strong short
    if fs <= STRONG_SHORT:
        return ["background-color: rgba(140, 0, 0, 0.28);"] * len(row)
    return [""] * len(row)

display_cols = ["SOURCE","SYMBOL","FINAL_SCORE","SIGNAL","KUCOIN_SCORE","BINANCE_SCORE","PRICE","QV_24H","KAPI_KUCOIN","KAPI_BINANCE","DUAL_CONFIRMED","STRONG"]
df_show = df[display_cols].copy()

st.dataframe(
    df_show.style.apply(style_rows, axis=1),
    use_container_width=True,
    height=650
)

# Optional: quick filters (without breaking anything)
with st.expander("Filtre (isteğe bağlı)"):
    only_both = st.checkbox("Sadece BOTH göster", value=False)
    only_strong = st.checkbox("Sadece STRONG göster", value=False)
    if only_both or only_strong:
        dff = df_show.copy()
        if only_both:
            dff = dff[dff["SOURCE"] == "Both"]
        if only_strong:
            dff = dff[dff["STRONG"] == True]
        st.dataframe(dff.style.apply(style_rows, axis=1), use_container_width=True, height=450)

# Auto refresh
if auto:
    st.caption(f"Otomatik yenileme: {refresh} saniye")
    time.sleep(int(refresh))
    st.rerun()
