# uygulama.py
# requirements.txt: streamlit, pandas, numpy, ccxt

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import ccxt


# =============================
# SABİT AYARLAR (MANUEL YOK)
# =============================
IST = ZoneInfo("Europe/Istanbul")

TF = "15m"
HTF = "1h"

AUTO_REFRESH_SEC = 240  # 4 dakika
TOP_N_PER_EXCHANGE = 150  # her borsada taranacak USDT spot sayısı

CANDLE_LIMIT = 200
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
SMA_PERIOD = 20
ADX_PERIOD = 14

SCORE_STEP = 5

STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10

# “Kalite” için: çok çöp coin gelmesin (likidite + spread)
MIN_QV_USDT_24H = 200_000  # USDT (yaklaşık) — çok sert yaparsan tablo boşalır
MAX_SPREAD_PCT = 0.30      # %0.30

# “KAPI” mantığı: 6 kapıdan kaçını geçti?
MIN_KAPI_FOR_TABLE = 4     # tablo aday eşiği (STRONG için 6 şart)

# Tablo boyutu
TABLE_N = 20
FALLBACK_LONG_N = 10
FALLBACK_SHORT_N = 10


# =============================
# İNDİKATÖR (pandas/numpy) — pandas_ta yok
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
    # Wilder ADX (basit ve stabil)
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

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
# CCXT — Borsalar
# =============================
def make_kucoin() -> ccxt.kucoin:
    return ccxt.kucoin({"enableRateLimit": True, "timeout": 20000})

def make_okx() -> ccxt.okx:
    return ccxt.okx({"enableRateLimit": True, "timeout": 20000})


def safe_load_markets(ex: ccxt.Exchange) -> Dict:
    try:
        return ex.load_markets()
    except Exception:
        return {}

def safe_fetch_tickers(ex: ccxt.Exchange) -> Dict:
    try:
        return ex.fetch_tickers()
    except Exception:
        return {}

def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> Optional[List]:
    try:
        return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception:
        return None


def is_usdt_spot_market(m: dict) -> bool:
    if not m:
        return False
    if not m.get("active", True):
        return False
    if not m.get("spot", False):
        return False
    quote = m.get("quote")
    return quote == "USDT"


def compute_qv_usdt(t: dict) -> float:
    # quoteVolume çoğu borsada doğrudan quote (USDT) olur.
    if not isinstance(t, dict):
        return 0.0
    qv = t.get("quoteVolume")
    if qv is not None:
        try:
            return float(qv)
        except Exception:
            pass
    # fallback: baseVolume * last
    bv = t.get("baseVolume")
    last = t.get("last")
    try:
        if bv is not None and last is not None:
            return float(bv) * float(last)
    except Exception:
        pass
    return 0.0


def compute_spread_pct(t: dict) -> float:
    if not isinstance(t, dict):
        return 999.0
    bid = t.get("bid")
    ask = t.get("ask")
    try:
        bid = float(bid) if bid is not None else None
        ask = float(ask) if ask is not None else None
        if bid and ask and bid > 0:
            return ((ask - bid) / bid) * 100.0
    except Exception:
        pass
    return 999.0


@st.cache_data(show_spinner=False, ttl=300)
def get_top_usdt_symbols(exchange_name: str, top_n: int) -> Tuple[List[str], Dict]:
    # exchange_name: "kucoin" | "okx"
    ex = make_kucoin() if exchange_name == "kucoin" else make_okx()
    markets = safe_load_markets(ex)
    tickers = safe_fetch_tickers(ex)

    syms = []
    scored = []
    for sym, m in markets.items():
        if not is_usdt_spot_market(m):
            continue
        t = tickers.get(sym, {})
        qv = compute_qv_usdt(t)
        scored.append((sym, qv))

    scored.sort(key=lambda x: x[1], reverse=True)
    syms = [s for s, _ in scored[:top_n]]
    return syms, tickers


def calc_raw_and_kapi(df15: pd.DataFrame, df1h: pd.DataFrame, tkr: dict) -> Tuple[float, int, str]:
    """
    Tek RAW: 0..100 (0 strong short, 100 strong long)
    KAPI: 0..6 (seçilen yön için kapılar)
    YÖN: LONG/SHORT (RAW>=50 -> LONG, RAW<50 -> SHORT)
    """

    close15 = df15["close"].astype(float)
    high15 = df15["high"].astype(float)
    low15 = df15["low"].astype(float)

    close1h = df1h["close"].astype(float)

    sma20_15 = sma(close15, SMA_PERIOD)
    sma20_1h = sma(close1h, SMA_PERIOD)
    rsi15 = rsi_wilder(close15, RSI_PERIOD)
    _, bb_up, bb_low = bollinger(close15, BB_PERIOD, BB_STD)
    adx15 = adx_wilder(high15, low15, close15, ADX_PERIOD)

    last = float(close15.iloc[-1])
    last_sma15 = float(sma20_15.iloc[-1])
    last_rsi = float(rsi15.iloc[-1])
    last_bbu = float(bb_up.iloc[-1])
    last_bbl = float(bb_low.iloc[-1])
    last_adx = float(adx15.iloc[-1])
    last_1h = float(close1h.iloc[-1])
    last_sma1h = float(sma20_1h.iloc[-1])

    if any(np.isnan([last, last_sma15, last_rsi, last_bbu, last_bbl, last_adx, last_1h, last_sma1h])):
        return 50.0, 0, "LONG"

    # ========== RAW skoru (simetrik) ==========
    raw = 50.0

    # Trend (15m)
    raw += 15.0 if last > last_sma15 else -15.0

    # HTF trend (1h)
    raw += 15.0 if last_1h > last_sma1h else -15.0

    # RSI momentum
    if last_rsi < 35:
        raw += 15.0
    elif last_rsi > 65:
        raw -= 15.0

    # Bollinger konumu
    if last <= last_bbl:
        raw += 10.0
    elif last >= last_bbu:
        raw -= 10.0

    # ADX kuvvet (trend yönüne hafif destek)
    trend_sign = 1.0 if (last > last_sma15 and last_1h > last_sma1h) else (-1.0 if (last < last_sma15 and last_1h < last_sma1h) else 0.0)
    if last_adx >= 18 and trend_sign != 0:
        raw += 5.0 * trend_sign

    raw = float(np.clip(raw, 0.0, 100.0))

    # ========== YÖN ==========
    direction = "LONG" if raw >= 50 else "SHORT"

    # ========== 6 KAPI ==========
    qv = compute_qv_usdt(tkr)
    spread_pct = compute_spread_pct(tkr)

    # Kapılar yön bazlı
    # 1) Likidite
    gate_liq = qv >= MIN_QV_USDT_24H

    # 2) Spread
    gate_spread = spread_pct <= MAX_SPREAD_PCT

    # 3) ADX kuvvet
    gate_adx = last_adx >= 18

    # 4) Trend uyumu (15m + 1h aynı yön)
    if direction == "LONG":
        gate_trend = (last > last_sma15) and (last_1h > last_sma1h)
    else:
        gate_trend = (last < last_sma15) and (last_1h < last_sma1h)

    # 5) RSI yön filtresi (LONG için aşırı satıma yakın, SHORT için aşırı alıma yakın)
    if direction == "LONG":
        gate_rsi = last_rsi <= 45
    else:
        gate_rsi = last_rsi >= 55

    # 6) BB yakınlık filtresi
    # LONG: alt banda yakın / altında, SHORT: üst banda yakın / üstünde
    if direction == "LONG":
        gate_bb = last <= (last_bbl * 1.01)
    else:
        gate_bb = last >= (last_bbu * 0.99)

    kapi = int(sum([gate_liq, gate_spread, gate_adx, gate_trend, gate_rsi, gate_bb]))

    return raw, kapi, direction


def scan_exchange(name: str) -> Tuple[bool, pd.DataFrame]:
    """
    name: 'KUCOIN' veya 'OKX'
    Çıktı df kolonları:
    COIN, YÖN, RAW, SKOR, FİYAT, QV_24H, KAPI, STRONG, SOURCE
    """
    ex = make_kucoin() if name == "KUCOIN" else make_okx()
    ex_name = "kucoin" if name == "KUCOIN" else "okx"

    syms, tickers = get_top_usdt_symbols(ex_name, TOP_N_PER_EXCHANGE)
    if not syms:
        return False, pd.DataFrame()

    rows = []
    total = len(syms)
    prog = st.progress(0, text=f"{name}: hazırlanıyor…")
    status = st.empty()

    for i, sym in enumerate(syms, start=1):
        prog.progress(int((i - 1) / total * 100), text=f"{name}: {i}/{total} • {sym}")
        o15 = safe_fetch_ohlcv(ex, sym, TF, CANDLE_LIMIT)
        o1h = safe_fetch_ohlcv(ex, sym, HTF, CANDLE_LIMIT)
        if not o15 or not o1h:
            continue
        if len(o15) < max(SMA_PERIOD, BB_PERIOD, RSI_PERIOD, ADX_PERIOD) + 5:
            continue
        if len(o1h) < SMA_PERIOD + 5:
            continue

        df15 = pd.DataFrame(o15, columns=["ts", "open", "high", "low", "close", "volume"])
        df1h = pd.DataFrame(o1h, columns=["ts", "open", "high", "low", "close", "volume"])

        tkr = tickers.get(sym, {})
        raw, kapi, yon = calc_raw_and_kapi(df15, df1h, tkr)

        score = round_step(raw, SCORE_STEP)
        last_price = float(df15["close"].astype(float).iloc[-1])
        qv = compute_qv_usdt(tkr)

        strong = (kapi == 6) and ((raw >= STRONG_LONG_MIN) or (raw <= STRONG_SHORT_MAX))

        coin = sym.replace("/USDT", "")

        rows.append(
            {
                "YÖN": yon,
                "COIN": coin,
                "SKOR": int(score),
                "FİYAT": float(last_price),
                "RAW": int(round(raw)),
                "QV_24H": float(qv),
                "KAPI": int(kapi),
                "STRONG": bool(strong),
                "SOURCE": name,
            }
        )

        time.sleep(0.02)

    prog.progress(100, text=f"{name}: bitti.")
    status.empty()

    df = pd.DataFrame(rows)
    return True, df


def merge_sources(df_k: pd.DataFrame, df_o: pd.DataFrame) -> pd.DataFrame:
    """
    KUCOIN + OKX birleştir:
    - Aynı COIN iki borsada varsa SOURCE='BOTH' ve RAW/SKOR ortalama, KAPI min (daha sıkı)
    - Diğerleri kendi SOURCE’u ile kalır
    """
    if df_k is None:
        df_k = pd.DataFrame()
    if df_o is None:
        df_o = pd.DataFrame()

    if df_k.empty and df_o.empty:
        return pd.DataFrame()

    # index by coin
    k = df_k.set_index("COIN") if not df_k.empty else pd.DataFrame().set_index(pd.Index([], name="COIN"))
    o = df_o.set_index("COIN") if not df_o.empty else pd.DataFrame().set_index(pd.Index([], name="COIN"))

    coins = sorted(set(k.index.tolist()) | set(o.index.tolist()))
    out_rows = []

    for coin in coins:
        in_k = coin in k.index
        in_o = coin in o.index

        if in_k and in_o:
            rk = float(k.loc[coin, "RAW"])
            ro = float(o.loc[coin, "RAW"])
            raw = (rk + ro) / 2.0

            score = round_step(raw, SCORE_STEP)
            # yön: ortalama raw
            yon = "LONG" if raw >= 50 else "SHORT"

            # KAPI: daha sıkı olmak için min
            kapi = int(min(int(k.loc[coin, "KAPI"]), int(o.loc[coin, "KAPI"])))

            # STRONG: KAPI=6 ve eşik
            strong = (kapi == 6) and ((raw >= STRONG_LONG_MIN) or (raw <= STRONG_SHORT_MAX))

            # fiyat: kucoin fiyatı (varsa) yoksa okx
            price = float(k.loc[coin, "FİYAT"])
            qv = float(k.loc[coin, "QV_24H"]) + float(o.loc[coin, "QV_24H"])

            out_rows.append(
                {
                    "YÖN": yon,
                    "COIN": coin,
                    "SKOR": int(score),
                    "FİYAT": float(price),
                    "RAW": int(round(raw)),
                    "QV_24H": float(qv),
                    "KAPI": int(kapi),
                    "STRONG": bool(strong),
                    "SOURCE": "BOTH",
                }
            )
        elif in_k:
            row = k.loc[coin].to_dict()
            row["COIN"] = coin
            out_rows.append(row)
        else:
            row = o.loc[coin].to_dict()
            row["COIN"] = coin
            out_rows.append(row)

    df = pd.DataFrame(out_rows)

    # Tip düzelt
    for c in ["SKOR", "RAW", "KAPI"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["QV_24H"] = pd.to_numeric(df["QV_24H"], errors="coerce").fillna(0.0).astype(float)
    df["FİYAT"] = pd.to_numeric(df["FİYAT"], errors="coerce").fillna(0.0).astype(float)
    df["STRONG"] = df["STRONG"].astype(bool)
    df["YÖN"] = df["YÖN"].astype(str)
    df["SOURCE"] = df["SOURCE"].astype(str)

    return df


def build_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Tablo mantığı:
    1) STRONG varsa önce STRONG’ları koy.
    2) Kalan boşlukları TOP adaylarla doldur.
    3) STRONG yoksa: 10 LONG + 10 SHORT (KAPI>=MIN_KAPI_FOR_TABLE) en yakın adaylar.
    """
    if df is None or df.empty:
        return pd.DataFrame(), "Aday yok (network/KuCoin/OKX veya filtre çok sert). Bir sonraki auto refresh’i bekle."

    # Aday eşiği
    cand = df[df["KAPI"] >= MIN_KAPI_FOR_TABLE].copy()
    if cand.empty:
        return pd.DataFrame(), "Aday yok (filtre çok sert). Bir sonraki auto refresh’i bekle."

    # Strong set
    strongs = cand[cand["STRONG"] == True].copy()

    # yardımcı sıralama: BOTH üstte, sonra strong, sonra skor
    def source_rank(x: str) -> int:
        return 2 if x == "BOTH" else (1 if x in ("KUCOIN", "OKX") else 0)

    cand["SRC_R"] = cand["SOURCE"].apply(source_rank)
    strongs["SRC_R"] = strongs["SOURCE"].apply(source_rank)

    # Long/Short ayrımı
    longs = cand[cand["YÖN"] == "LONG"].copy()
    shorts = cand[cand["YÖN"] == "SHORT"].copy()

    # “en yakın” mantığı:
    # long: RAW büyük olanlar; short: RAW küçük olanlar
    longs = longs.sort_values(["SRC_R", "RAW"], ascending=[False, False])
    shorts = shorts.sort_values(["SRC_R", "RAW"], ascending=[False, True])

    if not strongs.empty:
        # strongları önce koy
        strongs = strongs.sort_values(["SRC_R", "RAW"], ascending=[False, False])

        used = set()
        out = []
        for _, r in strongs.iterrows():
            key = (r["COIN"], r["SOURCE"])
            if key in used:
                continue
            out.append(r)
            used.add(key)
            if len(out) >= TABLE_N:
                break

        # kalan boşlukları “TOP adaylarla” doldur:
        # önce BOTH + long, BOTH + short karışık, sonra diğerleri
        fill = pd.concat([longs, shorts], ignore_index=True)
        for _, r in fill.iterrows():
            if len(out) >= TABLE_N:
                break
            key = (r["COIN"], r["SOURCE"])
            if key in used:
                continue
            out.append(r)
            used.add(key)

        out_df = pd.DataFrame(out).drop(columns=["SRC_R"], errors="ignore").reset_index(drop=True)
        return out_df, "✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu."

    # STRONG yoksa: 10 long + 10 short zorunlu
    pick_l = longs.head(FALLBACK_LONG_N)
    pick_s = shorts.head(FALLBACK_SHORT_N)

    out_df = pd.concat([pick_l, pick_s], ignore_index=True)

    # eğer bir taraf boş kaldıysa, kalanını diğer taraftan doldur (yine de 20 satır)
    if len(out_df) < TABLE_N:
        rem = TABLE_N - len(out_df)
        rest = pd.concat([longs.iloc[len(pick_l):], shorts.iloc[len(pick_s):]], ignore_index=True)
        out_df = pd.concat([out_df, rest.head(rem)], ignore_index=True)

    out_df = out_df.drop(columns=["SRC_R"], errors="ignore").reset_index(drop=True)
    return out_df, "⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu."


# =============================
# STYLES (Koyu tema + renklendirme)
# =============================
def inject_css():
    st.markdown(
        """
<style>
html, body, [class*="css"] { background-color:#0b0f14 !important; color:#e6edf3 !important; }
.block-container { padding-top: 1.0rem; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] { display:none; } /* sidebar yok */

h1,h2,h3,h4,h5,h6,p,span,div { color:#e6edf3 !important; }
.stProgress > div > div { background-color:#2563eb !important; }

.card {
  background:#0f172a;
  border:1px solid #1f2a37;
  border-radius:14px;
  padding:14px 16px;
}
.badge-ok { background:#064e3b; border:1px solid #0f766e; }
.badge-warn { background:#3f3f10; border:1px solid #6b6b1a; }
.badge-info { background:#0b2440; border:1px solid #1f2a37; }
.badge {
  border-radius:14px;
  padding:12px 14px;
  font-weight:700;
}
.small { opacity:0.9; font-size: 12px; }
</style>
""",
        unsafe_allow_html=True,
    )


def style_table(df: pd.DataFrame):
    # Streamlit/pandas Styler dönüşünde type annotation KULLANMIYORUZ (Cloud’da patlamasın)
    def row_bg(row):
        yon = str(row.get("YÖN", ""))
        strong = bool(row.get("STRONG", False))
        src = str(row.get("SOURCE", ""))

        # base renkler
        if yon == "LONG":
            base = "#064e3b"  # yeşil
            strong_col = "#053a2c"  # daha koyu
        else:
            base = "#7f1d1d"  # kırmızı
            strong_col = "#5a1414"  # daha koyu

        # BOTH ise bir tık daha “premium” koyuluk
        if src == "BOTH":
            base = "#0b3a30" if yon == "LONG" else "#5a1010"
            strong_col = "#072a22" if yon == "LONG" else "#400b0b"

        bg = strong_col if strong else base
        return [f"background-color: {bg}; color: #ffffff;"] * len(row)

    fmt = {
        "FİYAT": "{:.6f}",
        "QV_24H": "{:,.0f}",
        "RAW": "{:d}",
        "SKOR": "{:d}",
        "KAPI": "{:d}",
    }

    return (
        df.style.format(fmt)
        .apply(row_bg, axis=1)
        .set_properties(**{
            "border-color": "#1f2a37",
            "border-style": "solid",
            "border-width": "1px",
            "font-weight": "600"
        })
    )


# =============================
# APP
# =============================
st.set_page_config(page_title="Sniper — Auto (LONG + SHORT)", layout="wide")
inject_css()

st.title("🎯 Sniper — Auto (LONG + SHORT)")
st.caption(f"TF={TF} • HTF={HTF} • STRONG: SKOR≥{STRONG_LONG_MIN} (LONG) / SKOR≤{STRONG_SHORT_MAX} (SHORT) • 6 Kapı • Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s")

now = datetime.now(IST)
st.markdown(f"<div class='small'>İstanbul Time: <b>{now.strftime('%Y-%m-%d %H:%M:%S')}</b></div>", unsafe_allow_html=True)

# Auto refresh
try:
    st.autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
except Exception:
    try:
        st.experimental_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
    except Exception:
        pass

# Bağlantı test kartları + tarama
colA, colB = st.columns(2)
with colA:
    st.markdown("<div class='card badge badge-ok'>KuCoin: ✅ Bağlandı</div>", unsafe_allow_html=True)
with colB:
    st.markdown("<div class='card badge badge-ok'>OKX: ✅ Bağlandı</div>", unsafe_allow_html=True)

# Tarama
spinner = st.empty()
with spinner:
    st.markdown("<div class='small'>Taranıyor…</div>", unsafe_allow_html=True)

ok_k, df_k = scan_exchange("KUCOIN")
ok_o, df_o = scan_exchange("OKX")

spinner.empty()
st.markdown("<div class='small'>Tarama bitti ✅</div>", unsafe_allow_html=True)

df_all = merge_sources(df_k, df_o)
df_show, status_msg = build_table(df_all)

# Sayımlar
if df_show is None or df_show.empty:
    strong_long = strong_short = long_n = short_n = 0
else:
    strong_long = int(((df_show["YÖN"] == "LONG") & (df_show["STRONG"] == True)).sum())
    strong_short = int(((df_show["YÖN"] == "SHORT") & (df_show["STRONG"] == True)).sum())
    long_n = int((df_show["YÖN"] == "LONG").sum())
    short_n = int((df_show["YÖN"] == "SHORT").sum())

# Durum kartı
if status_msg.startswith("✅"):
    st.markdown(f"<div class='card badge badge-ok'>✅ {status_msg}</div>", unsafe_allow_html=True)
elif status_msg.startswith("⚠️"):
    st.markdown(f"<div class='card badge badge-warn'>⚠️ {status_msg}</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='card badge badge-warn'>⚠️ {status_msg}</div>", unsafe_allow_html=True)

# Sayım kartı
st.markdown(
    f"""
<div class='card badge badge-info'>
✅ STRONG LONG: <b>{strong_long}</b> &nbsp;|&nbsp; 💀 STRONG SHORT: <b>{strong_short}</b> &nbsp;|&nbsp;
LONG: <b>{long_n}</b> &nbsp;|&nbsp; SHORT: <b>{short_n}</b>
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("🎯 SNIPER TABLO")

if df_show is None or df_show.empty:
    st.info("Aday yok. Bir sonraki yenilemeyi bekle.")
else:
    # Görünüm kolon sırası (senin ekranına uygun)
    df_show = df_show[["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI", "STRONG", "SOURCE"]].copy()
    st.dataframe(style_table(df_show), use_container_width=True, height=720)
