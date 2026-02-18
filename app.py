from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import ccxt

# =========================================================
# CONFIG (SABİT / MENÜ YOK)
# =========================================================
IST = ZoneInfo("Europe/Istanbul")

TIMEFRAME = "15m"
AUTO_REFRESH_SEC = 240

CANDLE_LIMIT = 220  # indikatörler için yeterli
UNIVERSE_MAX = 950  # "tüm coinler" pratikte KuCoin USDT spot aktiflerin tamamına yakın
MIN_QV24H = 20_000  # çok illiquid olanları ele (istersen 0 yaparsın ama tavsiye etmem)
MAX_SPREAD_PCT = 0.60  # %0.60 üstü spread: kaldıraçta can yakar

# STRONG eşiği
STRONG_LONG_RAW = 90
STRONG_SHORT_RAW = 10

# 6 Kapı - Seviye 2
MIN_GATES_SHOW = 2
MIN_GATES_STRONG = 4

# tablo kaç satır
TABLE_ROWS = 20

# puan adımı
SCORE_STEP = 5  # 1'er değil 5'er

# =========================================================
# INDICATORS (numpy/pandas)
# =========================================================
def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()

def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).mean()

def rsi_wilder(s: pd.Series, period: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0.0)
    loss = (-d).clip(lower=0.0)
    ag = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)

def bollinger(s: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = sma(s, period)
    std = s.rolling(period, min_periods=period).std(ddof=0)
    up = mid + n_std * std
    low = mid - n_std * std
    return mid, up, low

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

def macd_hist(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    m_fast = ema(s, fast)
    m_slow = ema(s, slow)
    m = m_fast - m_slow
    sig = ema(m, signal)
    return (m - sig).fillna(0.0)

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def round_step(x: float, step: int) -> int:
    return int(round(x / step) * step)

# =========================================================
# KUCOIN / CCXT
# =========================================================
def make_exchange() -> ccxt.kucoin:
    return ccxt.kucoin(
        {
            "enableRateLimit": True,
            "timeout": 20000,
        }
    )

@st.cache_data(show_spinner=False, ttl=900)
def load_usdt_spot_symbols() -> list[str]:
    ex = make_exchange()
    markets = ex.load_markets()
    syms = []
    for sym, m in markets.items():
        if not m:
            continue
        if not m.get("active", True):
            continue
        if not m.get("spot", False):
            continue
        if m.get("quote") != "USDT":
            continue
        syms.append(sym)
    return sorted(set(syms))

def safe_fetch_tickers(ex: ccxt.Exchange, symbols: list[str]) -> dict:
    # KuCoin bazen symbol listesiyle sıkıntı çıkarabilir → fallback all
    try:
        return ex.fetch_tickers(symbols)
    except Exception:
        try:
            all_t = ex.fetch_tickers()
            return {s: all_t.get(s) for s in symbols if s in all_t}
        except Exception:
            return {}

def qv24h_from_ticker(t: dict) -> float:
    if not t or not isinstance(t, dict):
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

def spread_pct_from_ticker(t: dict) -> float:
    if not t or not isinstance(t, dict):
        return 999.0
    bid = t.get("bid")
    ask = t.get("ask")
    if bid is None or ask is None:
        return 999.0
    try:
        bid = float(bid)
        ask = float(ask)
        if bid <= 0 or ask <= 0:
            return 999.0
        mid = (bid + ask) / 2.0
        return (ask - bid) / mid * 100.0
    except Exception:
        return 999.0

def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
    # 2 deneme + küçük delay
    for _ in range(2):
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except (ccxt.RequestTimeout, ccxt.NetworkError):
            time.sleep(0.2)
        except Exception:
            break
    return None

# =========================================================
# DARK THEME CSS (ŞEFFAFLIK YOK)
# =========================================================
st.set_page_config(page_title="KuCoin PRO Sniper — Auto (LONG+SHORT)", layout="wide")
st.markdown(
    """
<style>
/* Genel koyu tema */
.stApp { background-color: #0b0f14; color: #e6edf3; }
html, body { background-color: #0b0f14; color: #e6edf3; }

/* Başlık / yazılar */
h1,h2,h3,h4,h5,p,span,div { color: #e6edf3 !important; }

/* Kart görünümleri */
.kcard {
  background: #0f172a;
  border: 1px solid #1f2a37;
  border-radius: 14px;
  padding: 14px 16px;
}
.kok {
  background: #0a1222;
  border: 1px solid #1f2a37;
  border-radius: 14px;
  padding: 12px 14px;
}

/* Dataframe etrafı */
[data-testid="stDataFrame"]{
  background: #0b0f14 !important;
  border: 1px solid #1f2a37 !important;
  border-radius: 14px !important;
  overflow: hidden !important;
}

/* Scroll bar (opsiyonel) */
::-webkit-scrollbar { height: 10px; width: 10px; }
::-webkit-scrollbar-thumb { background: #243244; border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# SCORE + 6 KAPI
# RAW: 0..100 (long'a yakınlık)
# SKOR: RAW'ın 5'lik yuvarlanmış hali
# KAPI: 0..6 filtre sayısı
# =========================================================
def compute_raw_and_gates(
    close: float,
    ema50: float,
    ema200: float,
    rsi14: float,
    bb_mid: float,
    bb_up: float,
    bb_low: float,
    atr_pct: float,
    macd_h: float,
    qv24h: float,
    spread_pct: float,
) -> tuple[int, int]:
    # --- RAW ---
    raw = 50.0

    # Trend: close - ema50 (±20)
    if ema50 > 0:
        trend = (close - ema50) / ema50
        raw += clamp(np.tanh(trend * 18.0) * 20.0, -20.0, 20.0)

    # RSI: oversold long (+25), overbought short (-25)
    raw += clamp(((50.0 - rsi14) / 50.0) * 25.0, -25.0, 25.0)

    # BB konumu: alt banda yakınsa long (+20), üst banda yakınsa short (-20)
    rng = (bb_up - bb_low)
    if rng > 0:
        z = (close - bb_mid) / (rng / 2.0)  # -1..+1 civarı
        raw += clamp(-z * 20.0, -20.0, 20.0)

    # MACD hist: yön teyidi (±10)
    if close > 0:
        m = (macd_h / close) * 800.0
        raw += clamp(np.tanh(m) * 10.0, -10.0, 10.0)

    raw = clamp(raw, 0.0, 100.0)

    # --- Direction ---
    is_long = raw >= 50.0

    # --- 6 KAPI ---
    gates = 0

    # Gate-1: Likidite
    if qv24h >= MIN_QV24H:
        gates += 1

    # Gate-2: Spread
    if spread_pct <= MAX_SPREAD_PCT:
        gates += 1

    # Gate-3: ATR% (çok sakin veya aşırı spike istemiyoruz)
    if 0.25 <= atr_pct <= 6.50:
        gates += 1

    # Gate-4: Trend hizası (EMA50/EMA200 + fiyat)
    if is_long:
        if close >= ema50 and ema50 >= ema200:
            gates += 1
    else:
        if close <= ema50 and ema50 <= ema200:
            gates += 1

    # Gate-5: RSI bölgesi (LONG için <55 daha iyi, SHORT için >45 daha iyi)
    if is_long:
        if rsi14 <= 55.0:
            gates += 1
    else:
        if rsi14 >= 45.0:
            gates += 1

    # Gate-6: BB uçları / MACD teyit (en az biri)
    if is_long:
        if (close <= bb_mid) or (macd_h > 0):
            gates += 1
    else:
        if (close >= bb_mid) or (macd_h < 0):
            gates += 1

    return int(round(raw)), int(gates)

def direction_from_raw(raw: int) -> str:
    return "LONG" if raw >= 50 else "SHORT"

def score_from_raw(raw: int) -> int:
    return int(clamp(round_step(raw, SCORE_STEP), 0, 100))

def strong_from_raw_gates(raw: int, gates: int) -> bool:
    return gates >= MIN_GATES_STRONG and (raw >= STRONG_LONG_RAW or raw <= STRONG_SHORT_RAW)

def strength_key(raw: int) -> int:
    # 50'den uzaklık: ne kadar "uç" o kadar güçlü aday
    return int(abs(raw - 50))

# =========================================================
# STYLE (RENKLER GERİ GELDİ)
# =========================================================
def style_table(df: pd.DataFrame):
    def dir_style(v: str):
        if str(v) == "LONG":
            return "background-color:#0b3b2e;color:#e6edf3;font-weight:700;"
        if str(v) == "SHORT":
            return "background-color:#4a1212;color:#e6edf3;font-weight:700;"
        return ""

    def score_style(v):
        try:
            v = int(v)
        except Exception:
            return ""
        if v >= 90:
            return "background-color:#0a6b2b;color:#ffffff;font-weight:800;"
        if v >= 75:
            return "background-color:#0c4a3a;color:#ffffff;font-weight:700;"
        if v <= 10:
            return "background-color:#8b0000;color:#ffffff;font-weight:800;"
        if v <= 25:
            return "background-color:#5a1a1a;color:#ffffff;font-weight:700;"
        return ""

    fmt = {
        "FİYAT": "{:.6f}",
        "QV_24H": "{:,.0f}",
        "RAW": "{:d}",
        "SKOR": "{:d}",
        "KAPI": "{:d}",
    }

    return (
        df.style.format(fmt)
        .applymap(dir_style, subset=["YÖN"])
        .applymap(score_style, subset=["SKOR"])
        .set_table_styles(
            [
                {"selector": "th", "props": "background-color:#111827;color:#e6edf3;border:1px solid #1f2a37;"},
                {"selector": "td", "props": "background-color:#0b0f14;color:#e6edf3;border:1px solid #1f2a37;"},
            ]
        )
    )

# =========================================================
# MAIN SCAN
# =========================================================
def build_table() -> tuple[pd.DataFrame, dict]:
    ex = make_exchange()

    syms = load_usdt_spot_symbols()
    meta = {
        "universe": len(syms),
        "after_liq": 0,
        "scanned": 0,
        "strong_count": 0,
        "now": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not syms:
        return pd.DataFrame(), meta

    # tickers (likidite + spread için)
    tickers = safe_fetch_tickers(ex, syms)
    rows_rank = []
    for s in syms:
        t = tickers.get(s)
        qv = qv24h_from_ticker(t)
        spr = spread_pct_from_ticker(t)
        if qv >= MIN_QV24H:  # likidite filtresi
            rows_rank.append((s, qv, spr))
    rows_rank.sort(key=lambda x: x[1], reverse=True)

    meta["after_liq"] = len(rows_rank)

    # "tüm coinler" isteğine en yakın: likidite filtresinden geçenlerin tamamını tara,
    # ama aşırı uçta zaman uzarsa diye UNIVERSE_MAX ile güvenlik.
    scan_list = rows_rank[: min(len(rows_rank), UNIVERSE_MAX)]

    out = []
    total = len(scan_list)
    meta["scanned"] = total

    # Progress UI dışarıdan verilecek, burada sadece hesap
    for symbol, qv, spr in scan_list:
        ohlcv = safe_fetch_ohlcv(ex, symbol, TIMEFRAME, CANDLE_LIMIT)
        if not ohlcv or len(ohlcv) < 210:
            continue

        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        close_s = df["close"].astype(float)
        close = float(close_s.iloc[-1])

        ema50 = float(ema(close_s, 50).iloc[-1])
        ema200 = float(ema(close_s, 200).iloc[-1])
        rsi14 = float(rsi_wilder(close_s, 14).iloc[-1])
        bb_mid, bb_up, bb_low = bollinger(close_s, 20, 2.0)
        bbm = float(bb_mid.iloc[-1])
        bbu = float(bb_up.iloc[-1])
        bbl = float(bb_low.iloc[-1])
        atr14 = float(atr(df, 14).iloc[-1])
        atr_pct = (atr14 / close * 100.0) if close > 0 else 0.0
        mh = float(macd_hist(close_s, 12, 26, 9).iloc[-1])

        if any(np.isnan([ema50, ema200, rsi14, bbm, bbu, bbl, atr_pct, mh])):
            continue

        raw, gates = compute_raw_and_gates(
            close=close,
            ema50=ema50,
            ema200=ema200,
            rsi14=rsi14,
            bb_mid=bbm,
            bb_up=bbu,
            bb_low=bbl,
            atr_pct=atr_pct,
            macd_h=mh,
            qv24h=float(qv),
            spread_pct=float(spr),
        )

        # Seviye 2: en az 2 kapı geçmeden tabloya aday olmasın
        if gates < MIN_GATES_SHOW:
            continue

        skor = score_from_raw(raw)
        yon = direction_from_raw(raw)
        is_strong = strong_from_raw_gates(raw, gates)

        out.append(
            {
                "YÖN": yon,
                "COIN": symbol.replace("/USDT", ""),
                "SKOR": int(skor),
                "FİYAT": float(close),
                "RAW": int(raw),
                "QV_24H": float(qv),
                "KAPI": int(gates),
                "_strong": 1 if is_strong else 0,
                "_strength": strength_key(raw),
            }
        )

    if not out:
        return pd.DataFrame(), meta

    df_out = pd.DataFrame(out)

    # STRONG sayısı
    meta["strong_count"] = int(df_out["_strong"].sum())

    # 1) STRONG'ları öne al
    strong_df = df_out[df_out["_strong"] == 1].copy()

    # STRONG sıralama: LONG'larda RAW desc, SHORT'larda RAW asc, ayrıca KAPI desc, QV desc
    if not strong_df.empty:
        strong_df["_raw_sort"] = np.where(strong_df["YÖN"] == "LONG", strong_df["RAW"], 100 - strong_df["RAW"])
        strong_df = strong_df.sort_values(["KAPI", "_raw_sort", "QV_24H"], ascending=[False, False, False])

    # 2) Boş kalırsa TOP adaylarla doldur
    rest_df = df_out[df_out["_strong"] == 0].copy()
    # TOP aday sıralama: uçlara yakınlık (strength), sonra KAPI, sonra QV
    if not rest_df.empty:
        rest_df = rest_df.sort_values(["_strength", "KAPI", "QV_24H"], ascending=[False, False, False])

    final = strong_df
    if len(final) < TABLE_ROWS and not rest_df.empty:
        need = TABLE_ROWS - len(final)
        final = pd.concat([final, rest_df.head(need)], ignore_index=True)

    # Sütun düzeni
    final = final[["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI"]].reset_index(drop=True)
    return final, meta

# =========================================================
# AUTO REFRESH (240s)
# =========================================================
def try_autorefresh(interval_ms: int, key: str):
    try:
        return st.autorefresh(interval=interval_ms, key=key)
    except Exception:
        try:
            return st.experimental_autorefresh(interval=interval_ms, key=key)
        except Exception:
            return None

try_autorefresh(AUTO_REFRESH_SEC * 1000, "kucoin_pro_sniper_auto")

# =========================================================
# UI HEADER
# =========================================================
left, right = st.columns([2, 1], vertical_alignment="center")
with left:
    st.title("KuCoin PRO Sniper — Auto (LONG + SHORT)")
    st.caption(
        f"TF={TIMEFRAME} • STRONG: RAW≥{STRONG_LONG_RAW} LONG / RAW≤{STRONG_SHORT_RAW} SHORT • "
        f"Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s"
    )
with right:
    st.markdown(
        f"""
<div style="text-align:right; padding-top:6px;">
  <div style="font-size:12px; opacity:0.9;">Istanbul Time</div>
  <div style="font-size:18px; font-weight:800;">{datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# =========================================================
# SCAN + LOADING
# =========================================================
status_box = st.empty()
table_box = st.empty()

with st.spinner("⏳ KuCoin USDT spot evreni taranıyor... (lütfen bekle)"):
    # küçük bir ilerleme çubuğu: gerçek scan sayısı çok olunca en azından ekranda hareket görünsün
    prog = st.progress(0)
    # scan’i bir kerede yapıyoruz; progress’i basitçe time-based ilerletiyoruz
    # (ccxt rate-limit yüzünden gerçek adım adım takip bazen UI’ı kasıyor)
    start_t = time.time()
    df, meta = build_table()
    # hızlıca progress’i tamamla
    prog.progress(100)
    time.sleep(0.05)

# Üst bilgi kartları
status_box.markdown(
    f"""
<div class="kcard">
  <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
    <div style="font-weight:800;">🧠 Evren (USDT spot): {meta.get("universe","-")}</div>
    <div style="opacity:0.9;">• Likidite filtresi sonrası: {meta.get("after_liq","-")}</div>
    <div style="opacity:0.9;">• Tarama: {meta.get("scanned","-")}</div>
    <div style="opacity:0.9;">• Son: {meta.get("now","-")}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

st.subheader("🎯 SNIPER TABLO")

if df is None or df.empty:
    st.warning("Aday yok. (Network/KuCoin gecikmesi veya filtreler çok sert olabilir.) Bir sonraki auto refresh'i bekle.")
else:
    # STRONG mesajı
    strong_count = 0
    try:
        # strong sayısını meta’dan aldık
        strong_count = int(meta.get("strong_count", 0))
    except Exception:
        strong_count = 0

    if strong_count > 0:
        st.success("✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu.")
    else:
        st.warning("⚠️ Şu an STRONG yok. Tablo en iyi TOP adaylarla dolduruldu.")

    # Dataframe style
    sty = style_table(df)

    # use_container_width uyarısını tetiklememek için width='stretch'
    try:
        table_box.dataframe(sty, width="stretch", height=680)
    except TypeError:
        # eski streamlit uyumu
        table_box.dataframe(sty, use_container_width=True, height=680)

# Dip not
st.caption("RAW: 0–100 (long’a yakınlık) • SKOR: RAW’ın 5’lik adım yuvarlaması • KAPI: 6 filtreden kaçını geçti")
