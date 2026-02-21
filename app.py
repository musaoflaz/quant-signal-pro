# uygulama.py
from __future__ import annotations
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import ccxt
import requests
from streamlit_autorefresh import st_autorefresh

# =============================
# SABİT AYARLAR & TELEGRAM
# =============================
IST = ZoneInfo("Europe/Istanbul")
TG_TOKEN = "8330775219:AAHx20fZA6C3ONs5S8ELQrMpFEYba-bPN1k"
TG_CHAT_ID = "1358384022" 

TF = "15m"
HTF = "1h"
AUTO_REFRESH_SEC = 240
TOP_N_PER_EXCHANGE = 150
CANDLE_LIMIT = 200
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIOD = 20
ADX_PERIOD = 14
SCORE_STEP = 5
STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10
MIN_QV_USDT_24H = 200_000
MAX_SPREAD_PCT = 0.30
MIN_KAPI_FOR_TABLE = 4
TABLE_N = 20
FALLBACK_LONG_N = 10
FALLBACK_SHORT_N = 10

# =============================
# TELEGRAM FONKSİYONU
# =============================
def send_telegram_msg(message):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

def check_and_notify(df):
    if df is None or df.empty: return
    # Sadece BOTH (OKX+KuCoin) ve STRONG olan en kaliteli sinyalleri atar
    alerts = df[(df["SOURCE"] == "BOTH") & (df["STRONG"] == True)]
    for _, row in alerts.iterrows():
        emoji = "🚀 LONG" if row["YÖN"] == "LONG" else "💀 SHORT"
        msg = (
            f"🎯 <b>SNIPER SİNYAL</b>\n\n"
            f"<b>Coin:</b> #{row['COIN']}\n"
            f"<b>İşlem:</b> {emoji}\n"
            f"<b>Skor:</b> {row['SKOR']}\n"
            f"<b>Fiyat:</b> {row['FİYAT']}\n"
            f"<b>Borsa:</b> OKX + KuCoin Onaylı ✅"
        )
        send_telegram_msg(msg)

# =============================
# İNDİKATÖRLER (ORİJİNAL)
# =============================
def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).mean()

def bollinger(series: pd.Series, period: int, n_std: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower

def rsi_wilder(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_sm = pd.Series(tr).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    plus_sm = pd.Series(plus_dm, index=high.index).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    minus_sm = pd.Series(minus_dm, index=high.index).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * (plus_sm / tr_sm.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_sm / tr_sm.replace(0.0, np.nan))
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
    adx = dx.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)

def round_step(x: float, step: int) -> int:
    return int(np.clip(int(round(x / step) * step), 0, 100))

# =============================
# CCXT BORSALAR
# =============================
def make_kucoin() -> ccxt.kucoin:
    return ccxt.kucoin({"enableRateLimit": True, "timeout": 20000})

def make_okx() -> ccxt.okx:
    return ccxt.okx({"enableRateLimit": True, "timeout": 20000})

def safe_load_markets(ex: ccxt.Exchange) -> Dict:
    try: return ex.load_markets()
    except: return {}

def safe_fetch_tickers(ex: ccxt.Exchange) -> Dict:
    try: return ex.fetch_tickers()
    except: return {}

def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[List]:
    try: return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except: return None

def is_usdt_spot_market(m: dict) -> bool:
    if not m or not m.get("active", True) or not m.get("spot", False): return False
    return m.get("quote") == "USDT"

def compute_qv_usdt(t: dict) -> float:
    if not isinstance(t, dict): return 0.0
    qv = t.get("quoteVolume")
    if qv is not None:
        try: return float(qv)
        except: pass
    bv = t.get("baseVolume"); last = t.get("last")
    try:
        if bv and last: return float(bv) * float(last)
    except: pass
    return 0.0

def compute_spread_pct(t: dict) -> float:
    if not isinstance(t, dict): return 999.0
    bid = t.get("bid"); ask = t.get("ask")
    try:
        if bid and ask and bid > 0: return ((float(ask) - float(bid)) / float(bid)) * 100.0
    except: pass
    return 999.0

@st.cache_data(show_spinner=False, ttl=300)
def get_top_usdt_symbols(exchange_name: str, top_n: int) -> Tuple[List[str], Dict]:
    ex = make_kucoin() if exchange_name == "kucoin" else make_okx()
    markets = safe_load_markets(ex); tickers = safe_fetch_tickers(ex)
    scored = []
    for sym, m in markets.items():
        if not is_usdt_spot_market(m): continue
        qv = compute_qv_usdt(tickers.get(sym, {}))
        scored.append((sym, qv))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_n]], tickers

def calc_raw_and_kapi(df15: pd.DataFrame, df1h: pd.DataFrame, tkr: dict) -> Tuple[float, int, str]:
    close15 = df15["close"].astype(float); high15 = df15["high"].astype(float); low15 = df15["low"].astype(float)
    close1h = df1h["close"].astype(float)
    sma20_15 = sma(close15, SMA_PERIOD); sma20_1h = sma(close1h, SMA_PERIOD)
    rsi15 = rsi_wilder(close15, RSI_PERIOD); _, bb_up, bb_low = bollinger(close15, BB_PERIOD, BB_STD)
    adx15 = adx_wilder(high15, low15, close15, ADX_PERIOD)
    last = float(close15.iloc[-1]); last_sma15 = float(sma20_15.iloc[-1]); last_rsi = float(rsi15.iloc[-1])
    last_bbu = float(bb_up.iloc[-1]); last_bbl = float(bb_low.iloc[-1]); last_adx = float(adx15.iloc[-1])
    last_1h = float(close1h.iloc[-1]); last_sma1h = float(sma20_1h.iloc[-1])
    if any(np.isnan([last, last_sma15, last_rsi, last_bbu, last_bbl, last_adx, last_1h, last_sma1h])): return 50.0, 0, "LONG"
    raw = 50.0
    raw += 15.0 if last > last_sma15 else -15.0
    raw += 15.0 if last_1h > last_sma1h else -15.0
    if last_rsi < 35: raw += 15.0
    elif last_rsi > 65: raw -= 15.0
    if last <= last_bbl: raw += 10.0
    elif last >= last_bbu: raw -= 10.0
    trend_sign = 1.0 if (last > last_sma15 and last_1h > last_sma1h) else (-1.0 if (last < last_sma15 and last_1h < last_sma1h) else 0.0)
    if last_adx >= 18 and trend_sign != 0: raw += 5.0 * trend_sign
    raw = float(np.clip(raw, 0.0, 100.0))
    direction = "LONG" if raw >= 50 else "SHORT"
    qv = compute_qv_usdt(tkr); spread_pct = compute_spread_pct(tkr)
    gate_liq = qv >= MIN_QV_USDT_24H
    gate_spread = spread_pct <= MAX_SPREAD_PCT
    gate_adx = last_adx >= 18
    gate_trend = (last > last_sma15 and last_1h > last_sma1h) if direction == "LONG" else (last < last_sma15 and last_1h < last_sma1h)
    gate_rsi = (last_rsi <= 45) if direction == "LONG" else (last_rsi >= 55)
    gate_bb = (last <= last_bbl * 1.01) if direction == "LONG" else (last >= last_bbu * 0.99)
    kapi = int(sum([gate_liq, gate_spread, gate_adx, gate_trend, gate_rsi, gate_bb]))
    return raw, kapi, direction

def scan_exchange(name: str) -> Tuple[bool, pd.DataFrame]:
    ex = make_kucoin() if name == "KUCOIN" else make_okx()
    syms, tickers = get_top_usdt_symbols(name.lower(), TOP_N_PER_EXCHANGE)
    if not syms: return False, pd.DataFrame()
    rows = []
    total = len(syms)
    prog = st.progress(0, text=f"{name}: taranıyor...")
    for i, sym in enumerate(syms, start=1):
        prog.progress(int(i / total * 100))
        o15 = safe_fetch_ohlcv(ex, sym, TF, CANDLE_LIMIT)
        o1h = safe_fetch_ohlcv(ex, sym, HTF, CANDLE_LIMIT)
        if not o15 or not o1h or len(o15) < 30: continue
        df15 = pd.DataFrame(o15, columns=["ts", "open", "high", "low", "close", "volume"])
        df1h = pd.DataFrame(o1h, columns=["ts", "open", "high", "low", "close", "volume"])
        tkr = tickers.get(sym, {})
        raw, kapi, yon = calc_raw_and_kapi(df15, df1h, tkr)
        score = round_step(raw, SCORE_STEP)
        rows.append({
            "YÖN": yon, "COIN": sym.replace("/USDT", ""), "SKOR": int(score),
            "FİYAT": float(df15["close"].iloc[-1]), "RAW": int(round(raw)),
            "QV_24H": float(compute_qv_usdt(tkr)), "KAPI": int(kapi),
            "STRONG": (kapi == 6 and (raw >= STRONG_LONG_MIN or raw <= STRONG_SHORT_MAX)),
            "SOURCE": name
        })
        time.sleep(0.01)
    prog.empty()
    return True, pd.DataFrame(rows)

def merge_sources(df_k, df_o):
    if (df_k is None or df_k.empty) and (df_o is None or df_o.empty): return pd.DataFrame()
    if df_k is None or df_k.empty: return df_o
    if df_o is None or df_o.empty: return df_k
    k = df_k.set_index("COIN"); o = df_o.set_index("COIN")
    coins = sorted(set(k.index) | set(o.index))
    out = []
    for c in coins:
        if c in k.index and c in o.index:
            raw = (k.loc[c, "RAW"] + o.loc[c, "RAW"]) / 2
            out.append({
                "YÖN": "LONG" if raw >= 50 else "SHORT", "COIN": c, "SKOR": round_step(raw, SCORE_STEP),
                "FİYAT": k.loc[c, "FİYAT"], "RAW": int(round(raw)), "QV_24H": k.loc[c, "QV_24H"] + o.loc[c, "QV_24H"],
                "KAPI": min(k.loc[c, "KAPI"], o.loc[c, "KAPI"]), "SOURCE": "BOTH",
                "STRONG": (min(k.loc[c, "KAPI"], o.loc[c, "KAPI"]) == 6 and (raw >= STRONG_LONG_MIN or raw <= STRONG_SHORT_MAX))
            })
        elif c in k.index: out.append(k.loc[c].to_dict() | {"COIN": c})
        else: out.append(o.loc[c].to_dict() | {"COIN": c})
    return pd.DataFrame(out)

def build_table(df):
    if df is None or df.empty: return pd.DataFrame(), "Aday yok."
    cand = df[df["KAPI"] >= MIN_KAPI_FOR_TABLE].copy()
    if cand.empty: return pd.DataFrame(), "Aday yok."
    cand["SRC_R"] = cand["SOURCE"].apply(lambda x: 2 if x == "BOTH" else 1)
    strongs = cand[cand["STRONG"] == True].sort_values(["SRC_R", "RAW"], ascending=[False, False])
    longs = cand[cand["YÖN"] == "LONG"].sort_values(["SRC_R", "RAW"], ascending=[False, False])
    shorts = cand[cand["YÖN"] == "SHORT"].sort_values(["SRC_R", "RAW"], ascending=[False, True])
    res = pd.concat([strongs, longs.head(FALLBACK_LONG_N), shorts.head(FALLBACK_SHORT_N)]).drop_duplicates("COIN").head(TABLE_N)
    return res, "✅ Tarama Tamamlandı."

# =============================
# STREAMLIT UI & RUN
# =============================
st.set_page_config(page_title="Sniper OKX+KUCOIN", layout="wide")
st.title("🎯 Sniper — OKX & KuCoin")

st_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="refresh")

ok_k, df_k = scan_exchange("KUCOIN")
ok_o, df_o = scan_exchange("OKX")

df_all = merge_sources(df_k, df_o)
df_show, status = build_table(df_all)

# TELEGRAM BİLDİRİM TETİKLEME
check_and_notify(df_all)

st.info(status)
if not df_show.empty:
    st.dataframe(df_show[["YÖN", "COIN", "SKOR", "FİYAT", "KAPI", "SOURCE", "STRONG"]], use_container_width=True, height=600)
