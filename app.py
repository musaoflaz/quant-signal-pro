# app.py
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
# SABİT AYARLAR (MENÜ YOK)
# =============================
IST_TZ = ZoneInfo("Europe/Istanbul")

TIMEFRAME = "15m"
CANDLE_LIMIT = 200

AUTO_REFRESH_SEC = 240

# Evren tarama (USDT spot içinden, likiditeye göre ilk N)
UNIVERSE_MAX = 900

# Tablo satırı
TABLE_N = 20

# STRONG eşikleri
STRONG_LONG_RAW = 90
STRONG_SHORT_RAW = 10

# Skor adımı (5'er)
SCORE_STEP = 5

# 6 KAPI / SEVİYE 2
GATE_LEVEL = 2  # 1 = daha kolay, 2 = ideal (6 kapı)

# Kapı eşikleri (Seviye 2)
MIN_QV_USDT_24H = 20_000  # minimum 24h quoteVolume
MAX_SPREAD_PCT = 0.6      # %0.6 üstü spread proxy ise ele (illiquid)
MIN_ATR_PCT = 0.25        # çok düşük volatilite ele
MAX_ATR_PCT = 9.0         # aşırı spike ele
MIN_VOL_SPIKE = 1.15      # son hacim / ort hacim >= 1.15
MIN_ADX = 15.0            # trend gücü en az 15 (çok zayıf trend ele)


# =============================
# STREAMLIT TEMEL
# =============================
st.set_page_config(page_title="KuCoin PRO Sniper — Auto (LONG + SHORT)", layout="wide")

# Koyu tema + okunurluk (şeffaflık / soluk yazı yok)
st.markdown(
    """
<style>
/* App background */
[data-testid="stAppViewContainer"] { background: #0b0f14 !important; }
html, body, [class*="css"] { background: #0b0f14 !important; }

/* Container spacing */
.block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }

/* Text colors */
h1,h2,h3,h4,h5,h6,p,span,div,label { color: #e6edf3 !important; }

/* Remove header bg */
[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }

/* Cards */
.sniper-card {
  background: #0f172a;
  border: 1px solid #1f2a37;
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.sniper-warn {
  background: #1b1f10;
  border: 1px solid #3a3f1c;
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.sniper-ok {
  background: #0f2a1a;
  border: 1px solid #1f6b3a;
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 12px;
}

/* Make dataframe background dark-ish (styler does most) */
[data-testid="stDataFrame"] { background: #0b0f14 !important; }

</style>
""",
    unsafe_allow_html=True,
)


# =============================
# CCXT HELPERS
# =============================
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


def safe_fetch_tickers(ex: ccxt.Exchange) -> dict:
    try:
        return ex.fetch_tickers()
    except Exception:
        return {}


def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


# =============================
# INDICATORS (PURE NUMPY/PANDAS)
# =============================
def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi_wilder(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr


def atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(h, l, c)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def macd(c: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(c, fast)
    slow_ema = ema(c, slow)
    line = fast_ema - slow_ema
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def adx(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    up = h.diff()
    down = -l.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = true_range(h, l, c)

    atr_s = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=h.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_s.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=h.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr_s.replace(0.0, np.nan)

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx_s = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx_s.fillna(0.0)


def bollinger(c: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = c.rolling(period, min_periods=period).mean()
    std = c.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower


# =============================
# SCORING
# RAW: 0..100 (LONG bias)
# SKOR: 5'er adım
# =============================
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def quantize_score(raw: float, step: int = 5) -> int:
    raw = float(max(0.0, min(100.0, raw)))
    q = int(round(raw / step) * step)
    return int(max(0, min(100, q)))


def compute_raw_score(
    close: float,
    ema20: float,
    ema50: float,
    rsi14: float,
    macd_hist: float,
    adx14: float,
    bb_mid: float,
    bb_up: float,
    bb_low: float,
    atr_pct: float,
    vol_spike: float,
) -> float:
    """
    RAW = 0..100  (100 = çok güçlü LONG bias)
    0   = çok güçlü SHORT bias
    50  = nötr
    """

    # 1) Trend bileşeni (0..1)
    trend = 0.5
    if ema50 > 0:
        trend = 0.5 + 0.5 * np.tanh((close - ema50) / (ema50 * 0.01))  # ~%1 ölçek

    # 2) Momentum (RSI) (0..1)
    # Long için RSI düşükse +, Short için RSI yüksekse -
    # 50 çevresi nötr
    rsi_comp = 0.5 + 0.5 * np.tanh((50.0 - rsi14) / 10.0) * (-1.0)  # RSI>50 -> long lehine değil
    # Yukarıdaki formülü daha anlaşılır hale getir:
    # RSI < 50 => rsi_comp > 0.5 (long lehine)
    # RSI > 50 => rsi_comp < 0.5 (short lehine)
    rsi_comp = float(rsi_comp)

    # 3) MACD histogram (0..1)
    macd_comp = 0.5 + 0.5 * np.tanh(macd_hist / (abs(macd_hist) + 1e-9 + 0.001))

    # 4) Bollinger pozisyonu (0..1)
    bb_comp = 0.5
    if (bb_up - bb_low) > 0:
        pos = (close - bb_low) / (bb_up - bb_low)  # 0..1
        bb_comp = float(max(0.0, min(1.0, pos)))

    # 5) Trend gücü ADX (0..1)
    adx_comp = clamp01((adx14 - 10.0) / 25.0)  # 10->0, 35->1

    # 6) Vol/Volume bonus (0..1) — çok agresif değil, sadece ince ayar
    atr_comp = 1.0 - clamp01(abs(atr_pct - 2.5) / 6.0)  # ideal ~%2.5
    vol_comp = clamp01((vol_spike - 1.0) / 1.5)        # 1.0->0, 2.5->1

    # Ağırlıklar (toplam 1.0)
    w_trend = 0.28
    w_rsi = 0.22
    w_macd = 0.18
    w_bb = 0.14
    w_adx = 0.10
    w_micro = 0.08  # atr+vol

    micro = 0.5 * atr_comp + 0.5 * vol_comp

    # 0..1 skala
    score01 = (
        w_trend * trend +
        w_rsi * rsi_comp +
        w_macd * macd_comp +
        w_bb * bb_comp +
        w_adx * adx_comp +
        w_micro * micro
    )

    raw = 100.0 * float(max(0.0, min(1.0, score01)))
    return raw


# =============================
# 6 KAPI / LEVEL 2
# =============================
def compute_gates(
    direction: str,
    qv_24h: float,
    spread_pct: float,
    atr_pct: float,
    adx14: float,
    vol_spike: float,
    close: float,
    ema50: float,
    macd_hist: float,
) -> tuple[int, dict]:
    """
    6 kapı:
    1) Likidite
    2) Spread proxy
    3) ATR% aralığı
    4) ADX trend gücü
    5) Volume spike
    6) Trend yön uyumu (EMA50) + MACD yön uyumu (mini doğrulama)
    """
    details = {}

    g1 = qv_24h >= MIN_QV_USDT_24H
    details["liq"] = g1

    g2 = (spread_pct >= 0.0) and (spread_pct <= MAX_SPREAD_PCT)
    details["spr"] = g2

    g3 = (atr_pct >= MIN_ATR_PCT) and (atr_pct <= MAX_ATR_PCT)
    details["atr"] = g3

    g4 = adx14 >= MIN_ADX
    details["adx"] = g4

    g5 = vol_spike >= MIN_VOL_SPIKE
    details["vol"] = g5

    # yön uyumu: LONG -> close>ema50 ve macd_hist>=0 | SHORT -> close<ema50 ve macd_hist<=0
    if direction == "LONG":
        g6 = (close > ema50) and (macd_hist >= 0)
    else:
        g6 = (close < ema50) and (macd_hist <= 0)
    details["dir"] = g6

    passed = int(g1) + int(g2) + int(g3) + int(g4) + int(g5) + int(g6)
    return passed, details


# =============================
# BTC/ETH REGIME (bilgi amaçlı)
# =============================
def detect_regime(ex: ccxt.Exchange) -> str:
    """
    Bilgi amaçlı: BTC/ETH kısa trendine göre bias.
    Tabloyu KAPATMAZ, sadece banner yazar.
    """
    try:
        syms = ["BTC/USDT", "ETH/USDT"]
        bias = []
        for s in syms:
            o = safe_fetch_ohlcv(ex, s, "1h", 200)
            if not o or len(o) < 120:
                return "NEUTRAL"
            df = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "vol"])
            c = df["close"].astype(float)
            e50 = ema(c, 50).iloc[-1]
            e200 = ema(c, 200).iloc[-1]
            r = rsi_wilder(c, 14).iloc[-1]
            if (e50 > e200) and (r > 50):
                bias.append("BULL")
            elif (e50 < e200) and (r < 50):
                bias.append("BEAR")
            else:
                bias.append("NEUTRAL")

        if bias.count("BULL") == 2:
            return "LONG BIAS"
        if bias.count("BEAR") == 2:
            return "SHORT BIAS"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


# =============================
# TABLO STYLER
# =============================
def style_table(df: pd.DataFrame):
    def dir_style(v):
        if str(v) == "LONG":
            return "background-color:#0b3d2e; color:#ffffff; font-weight:700;"
        if str(v) == "SHORT":
            return "background-color:#5c1515; color:#ffffff; font-weight:700;"
        return ""

    def score_style(v):
        try:
            v = float(v)
        except Exception:
            return ""
        if v >= STRONG_LONG_RAW:
            return "background-color:#006400; color:#ffffff; font-weight:700;"
        if v <= STRONG_SHORT_RAW:
            return "background-color:#8B0000; color:#ffffff; font-weight:700;"
        return "background-color:#0f172a; color:#e6edf3;"

    fmt = {
        "FİYAT": "{:.6f}",
        "RAW": "{:.0f}",
        "QV_24H": "{:,.0f}",
        "KAPI": "{:d}",
    }

    return (
        df.style
        .format(fmt)
        .applymap(dir_style, subset=["YÖN"])
        .applymap(score_style, subset=["SKOR"])
        .set_properties(**{"border-color": "#1f2a37", "color": "#e6edf3", "background-color": "#0b0f14"})
    )


# =============================
# ANA TARAMA
# =============================
def build_universe(symbols: list[str], tickers: dict) -> list[tuple[str, float]]:
    rows = []
    for s in symbols:
        t = tickers.get(s) or {}
        qv = t.get("quoteVolume")
        last = t.get("last")
        if qv is None or last is None:
            continue
        try:
            qv = float(qv)
            last = float(last)
        except Exception:
            continue
        if last <= 0:
            continue
        rows.append((s, qv))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:UNIVERSE_MAX]


def compute_spread_pct(ticker: dict) -> float:
    """
    KuCoin tickers bazen bid/ask verir. Yoksa spread ölçemeyiz -> 0.0 (kapı 2'yi geçirmez hale getirmiyoruz)
    Ama illiquid'i yakalamak için mümkün olduğunda kullanıyoruz.
    """
    try:
        bid = ticker.get("bid")
        ask = ticker.get("ask")
        if bid is None or ask is None:
            return 0.0
        bid = float(bid)
        ask = float(ask)
        if bid <= 0 or ask <= 0:
            return 0.0
        mid = (bid + ask) / 2.0
        return 100.0 * (ask - bid) / mid
    except Exception:
        return 0.0


def scan_all() -> tuple[pd.DataFrame, dict]:
    ex = make_exchange()

    # 1) symbols + tickers
    syms = load_usdt_spot_symbols()
    tickers = safe_fetch_tickers(ex)

    universe = build_universe(syms, tickers)
    total = len(universe)

    meta = {
        "universe_total": len(syms),
        "universe_after_liq": total,
        "scanned": 0,
        "regime": detect_regime(ex),
    }

    rows = []
    gate_pass_map: dict[str, int] = {}   # <- BURASI kesin düzgün (SyntaxError yok)
    gate_detail_map: dict[str, dict] = {}

    prog = st.progress(0, text="⏳ KuCoin USDT spot evreni taranıyor...")

    for i, (symbol, qv) in enumerate(universe, start=1):
        meta["scanned"] = i
        if i % 10 == 0 or i == 1:
            prog.progress(int((i / max(1, total)) * 100), text=f"⏳ Taranıyor: {symbol} ({i}/{total})")

        t = tickers.get(symbol) or {}
        last = t.get("last")
        if last is None:
            continue
        try:
            last = float(last)
        except Exception:
            continue
        if last <= 0:
            continue

        spread_pct = compute_spread_pct(t)

        # ohlcv
        try:
            o = safe_fetch_ohlcv(ex, symbol, TIMEFRAME, CANDLE_LIMIT)
        except Exception:
            continue
        if not o or len(o) < 120:
            continue

        df = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "vol"])
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        v = df["vol"].astype(float)

        # indicators
        e20 = ema(c, 20)
        e50 = ema(c, 50)
        r14 = rsi_wilder(c, 14)
        _, _, mh = macd(c, 12, 26, 9)
        a14 = adx(h, l, c, 14)
        at = atr(h, l, c, 14)
        bbm, bbu, bbl = bollinger(c, 20, 2.0)

        # last values
        close = float(c.iloc[-1])
        ema20_v = float(e20.iloc[-1])
        ema50_v = float(e50.iloc[-1])
        rsi_v = float(r14.iloc[-1])
        macd_hist_v = float(mh.iloc[-1])
        adx_v = float(a14.iloc[-1])
        atr_v = float(at.iloc[-1])
        bbm_v = float(bbm.iloc[-1])
        bbu_v = float(bbu.iloc[-1])
        bbl_v = float(bbl.iloc[-1])

        # NaN kontrol
        if any(np.isnan([ema20_v, ema50_v, rsi_v, macd_hist_v, adx_v, atr_v, bbm_v, bbu_v, bbl_v])):
            continue

        atr_pct = 100.0 * (atr_v / close) if close > 0 else 0.0
        vol_ma = float(v.rolling(20, min_periods=20).mean().iloc[-1]) if len(v) >= 20 else float(v.mean())
        vol_spike = float(v.iloc[-1] / vol_ma) if vol_ma and vol_ma > 0 else 1.0

        raw = compute_raw_score(
            close=close,
            ema20=ema20_v,
            ema50=ema50_v,
            rsi14=rsi_v,
            macd_hist=macd_hist_v,
            adx14=adx_v,
            bb_mid=bbm_v,
            bb_up=bbu_v,
            bb_low=bbl_v,
            atr_pct=atr_pct,
            vol_spike=vol_spike,
        )

        direction = "LONG" if raw >= 50 else "SHORT"
        # 6 kapı
        gate_passed, gate_details = compute_gates(
            direction=direction,
            qv_24h=float(qv),
            spread_pct=float(spread_pct),
            atr_pct=float(atr_pct),
            adx14=float(adx_v),
            vol_spike=float(vol_spike),
            close=float(close),
            ema50=float(ema50_v),
            macd_hist=float(macd_hist_v),
        )

        gate_pass_map[symbol] = gate_passed
        gate_detail_map[symbol] = gate_details

        score = quantize_score(raw, SCORE_STEP)

        rows.append(
            {
                "YÖN": direction,
                "COIN": symbol.split("/")[0],
                "SKOR": score,
                "FİYAT": close,
                "RAW": raw,
                "QV_24H": float(qv),
                "KAPI": int(gate_passed),
            }
        )

        # minik rate-limit nazı
        time.sleep(0.02)

    prog.empty()

    out = pd.DataFrame(rows)
    return out, meta


def pick_table(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Tablo kuralı:
    1) STRONG varsa önce STRONG'ları koy
    2) Boş kalırsa TOP adaylarla doldur (LONG+SHORT birlikte)
       - "en yakın" gerçekten en yakın: LONG tarafı RAW->90'a yakın, SHORT tarafı RAW->10'a yakın.
    3) Level2: STRONG için 6/6 kapı şart.
       Fallback adaylarda kapı sayısı yüksek olanlar öne geçer.
    """
    if df is None or df.empty:
        return df, "⚠️ Veri yok (network/KuCoin)."

    # Level2 şartları
    if GATE_LEVEL == 2:
        strong_long = df[(df["RAW"] >= STRONG_LONG_RAW) & (df["KAPI"] >= 6) & (df["YÖN"] == "LONG")].copy()
        strong_short = df[(df["RAW"] <= STRONG_SHORT_RAW) & (df["KAPI"] >= 6) & (df["YÖN"] == "SHORT")].copy()
    else:
        strong_long = df[(df["RAW"] >= STRONG_LONG_RAW) & (df["YÖN"] == "LONG")].copy()
        strong_short = df[(df["RAW"] <= STRONG_SHORT_RAW) & (df["YÖN"] == "SHORT")].copy()

    strong = pd.concat([strong_long, strong_short], ignore_index=True)

    # STRONG sıralama: LONG raw desc, SHORT raw asc
    if not strong.empty:
        strong["_rank"] = np.where(strong["YÖN"] == "LONG", -strong["RAW"], strong["RAW"])
        strong = strong.sort_values(["_rank", "KAPI"], ascending=[True, False]).drop(columns=["_rank"])

    remain = TABLE_N - len(strong)
    if remain <= 0:
        msg = f"✅ STRONG bulundu. En güçlü {min(TABLE_N, len(strong))} sinyal gösteriliyor."
        return strong.head(TABLE_N).reset_index(drop=True), msg

    # Fallback aday havuzu: (STRONG olmayanlar)
    cand = df.copy()
    # Strong satırlarını çıkar (aynı coin tekrarlanmasın)
    if not strong.empty:
        used = set(strong["COIN"].tolist())
        cand = cand[~cand["COIN"].isin(list(used))].copy()

    # "en yakın" ölçüsü (kapı sayısı yüksek olan önde)
    cand["DIST"] = np.where(
        cand["YÖN"] == "LONG",
        (STRONG_LONG_RAW - cand["RAW"]).abs(),
        (cand["RAW"] - STRONG_SHORT_RAW).abs(),
    )

    # Seviye2: fallback'te de en az 4 kapı (illiquid çöpler düşsün)
    if GATE_LEVEL == 2:
        cand = cand[cand["KAPI"] >= 4].copy()

    # LONG ve SHORT dengeleyelim: 10 + 10 (mümkünse)
    need_long = max(0, min(10, remain))
    need_short = max(0, min(10, remain - need_long))

    cand_long = cand[cand["YÖN"] == "LONG"].sort_values(["KAPI", "DIST", "RAW"], ascending=[False, True, False]).head(need_long)
    cand_short = cand[cand["YÖN"] == "SHORT"].sort_values(["KAPI", "DIST", "RAW"], ascending=[False, True, True]).head(need_short)

    fill = pd.concat([cand_long, cand_short], ignore_index=True)

    # Eğer yine boş kaldıysa genel TOP ile doldur
    remain2 = remain - len(fill)
    if remain2 > 0:
        used2 = set(fill["COIN"].tolist())
        cand2 = cand[~cand["COIN"].isin(list(used2))].copy()
        cand2 = cand2.sort_values(["KAPI", "DIST"], ascending=[False, True]).head(remain2)
        fill = pd.concat([fill, cand2], ignore_index=True)

    final = pd.concat([strong, fill], ignore_index=True)

    if strong.empty:
        msg = "⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu."
    else:
        msg = f"✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu."

    # Son tablo: önce STRONG, sonra TOP; ama estetik için küçük bir sıralama:
    # STRONG zaten üstte. TOP içinde kapı yüksek ve en yakın olan üstte.
    if len(strong) < len(final):
        top_part = final.iloc[len(strong):].copy()
        top_part = top_part.sort_values(["KAPI", "DIST"], ascending=[False, True])
        final = pd.concat([strong, top_part], ignore_index=True)

    # Temizlik
    if "DIST" in final.columns:
        final = final.drop(columns=["DIST"], errors="ignore")

    return final.head(TABLE_N).reset_index(drop=True), msg


# =============================
# UI
# =============================
st.title("KuCoin PRO Sniper — Auto (LONG + SHORT)")
st.caption(f"TF={TIMEFRAME} • STRONG: RAW≥{STRONG_LONG_RAW} LONG / RAW≤{STRONG_SHORT_RAW} SHORT • Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s")

# auto refresh
try:
    st.autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh_sniper")
except Exception:
    try:
        st.experimental_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh_sniper")
    except Exception:
        pass

now_ist = datetime.now(IST_TZ)
st.markdown(
    f"""
<div style="text-align:right; margin-top:-8px; margin-bottom:8px;">
  <div style="font-size:12px; opacity:0.95;">Istanbul Time</div>
  <div style="font-size:18px; font-weight:800;">{now_ist.strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
""",
    unsafe_allow_html=True,
)

# Scan (always)
placeholder = st.empty()
with placeholder:
    st.markdown('<div class="sniper-card">⏳ KuCoin USDT spot evreni taranıyor... (auto refresh açık)</div>', unsafe_allow_html=True)

with st.spinner("⏳ Tarama yapılıyor..."):
    df_all, meta = scan_all()

placeholder.empty()

# Regime banner
regime = meta.get("regime", "NEUTRAL")
if regime == "LONG BIAS":
    st.markdown(f'<div class="sniper-ok">✅ REGIME: LONG BIAS • BTC/ETH bullish</div>', unsafe_allow_html=True)
elif regime == "SHORT BIAS":
    st.markdown(f'<div class="sniper-warn">⛔ REGIME: SHORT BIAS • BTC/ETH bearish</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="sniper-warn">⚠️ REGIME: NEUTRAL • BTC/ETH neutral (bias yok)</div>', unsafe_allow_html=True)

# Meta info card
st.markdown(
    f"""
<div class="sniper-card">
🧠 Evren (USDT spot): {meta.get("universe_total", 0):,} • Likidite filtresi sonrası: {meta.get("universe_after_liq", 0):,} • Tarama: {meta.get("scanned", 0):,}
</div>
""",
    unsafe_allow_html=True,
)

st.subheader("🎯 SNIPER TABLO")

df_pick, banner = pick_table(df_all)

# Banner
if banner.startswith("✅"):
    st.markdown(f'<div class="sniper-ok">{banner}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="sniper-warn">{banner}</div>', unsafe_allow_html=True)

if df_pick is None or df_pick.empty:
    st.markdown('<div class="sniper-warn">Aday yok. (Regime NO TRADE olabilir ya da network/KuCoin.) Bir sonraki yenilemede tekrar dene.</div>', unsafe_allow_html=True)
else:
    # sütun sırası
    show_cols = ["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI"]
    df_show = df_pick.loc[:, show_cols].copy()

    st.dataframe(style_table(df_show), use_container_width=True, height=720)
