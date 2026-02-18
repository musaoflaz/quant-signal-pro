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
IST = ZoneInfo("Europe/Istanbul")

TIMEFRAME = "15m"
CANDLE_LIMIT = 220
AUTO_REFRESH_SEC = 240

TABLE_SIZE = 20

# ✅ İSTEDİĞİN: STRONG 90+
STRONG_MIN_SCORE = 90

# Skor adımı: 5'er 5'er (100 yağmurunu azaltır)
SCORE_STEP = 5

# Evren çok büyük (900+). Cloud timeout yememek için likidite filtresi + cap
MIN_QUOTE_VOL_24H = 10_000  # USDT
MAX_SCAN_SYMBOLS = 450      # gerekirse yükseltiriz

# Ağırlıklar (toplam ~100)
W_RSI = 22
W_BB = 22
W_TREND = 18
W_MACD = 14
W_ADX = 10
W_ATR = 8
W_VOL = 6


# =============================
# KUCOIN / CCXT
# =============================
def make_exchange() -> ccxt.kucoin:
    return ccxt.kucoin(
        {
            "enableRateLimit": True,
            "timeout": 20000,
        }
    )


@st.cache_data(show_spinner=False, ttl=600)
def load_usdt_spot_symbols() -> list[str]:
    ex = make_exchange()
    markets = ex.load_markets()
    out: list[str] = []
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


def qv_24h(t: dict) -> float:
    if not t or not isinstance(t, dict):
        return 0.0
    qv = t.get("quoteVolume")
    if qv is not None:
        try:
            return float(qv)
        except Exception:
            return 0.0
    bv = t.get("baseVolume")
    last = t.get("last")
    try:
        if bv is not None and last is not None:
            return float(bv) * float(last)
    except Exception:
        pass
    return 0.0


# =============================
# İNDİKATÖRLER (PURE pandas/numpy)
# =============================
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    up = mid + k * std
    low = mid - k * std
    return mid, up, low


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ef = ema(close, fast)
    es = ema(close, slow)
    line = ef - es
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = true_range(high, low, close)
    atr_n = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_n

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().fillna(0.0)


# =============================
# SKORLAMA (LONG/SHORT ayrı, sonra max)
# =============================
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def quantize_score(x: float, step: int = 5) -> int:
    # 0..100 arası, step'e yuvarla
    x = clamp(x, 0, 100)
    q = int(round(x / step) * step)
    return int(clamp(q, 0, 100))


def score_one(df: pd.DataFrame) -> tuple[int, int, str]:
    """
    returns: (score_quantized, raw_best, direction)
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    if len(close) < 60:
        return 0, 0, "—"

    rsi = rsi_wilder(close, 14).iloc[-1]
    mid, up, low_bb = bollinger(close, 20, 2.0)
    sma20 = sma(close, 20)
    macd_line, macd_sig, macd_hist = macd(close)
    adx_v = adx(high, low, close, 14).iloc[-1]
    atr_v = atr(high, low, close, 14).iloc[-1]

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    last_sma = float(sma20.iloc[-1])
    prev_sma = float(sma20.iloc[-2])

    bb_up = float(up.iloc[-1])
    bb_low = float(low_bb.iloc[-1])
    bb_range = max(1e-9, (bb_up - bb_low))
    bb_pos = (last - bb_low) / bb_range  # 0..1

    hist = float(macd_hist.iloc[-1])

    # ATR% (çok oynaksa kaldıraçta risk)
    atr_pct = (atr_v / last) * 100.0 if last > 0 else 0.0

    # Volume spike (kırılım/ivme)
    vol_sma = float(sma(vol, 20).iloc[-1]) if len(vol) >= 20 else float(vol.mean())
    vol_ratio = (float(vol.iloc[-1]) / vol_sma) if vol_sma > 0 else 1.0

    # -------------- bileşen skorları --------------
    long_raw = 0.0
    short_raw = 0.0

    # RSI (low -> long, high -> short)
    # 25 altı çok güçlü long, 75 üstü çok güçlü short
    long_raw += W_RSI * clamp((50.0 - rsi) / 25.0, 0.0, 1.0)
    short_raw += W_RSI * clamp((rsi - 50.0) / 25.0, 0.0, 1.0)

    # Bollinger (alt banda yakın -> long, üst banda yakın -> short)
    long_raw += W_BB * clamp((0.35 - bb_pos) / 0.35, 0.0, 1.0)
    short_raw += W_BB * clamp((bb_pos - 0.65) / 0.35, 0.0, 1.0)

    # Trend (SMA20 üstü + SMA yükseliyor -> long, tersi -> short)
    sma_slope = last_sma - prev_sma
    if last > last_sma and sma_slope > 0:
        long_raw += W_TREND
    elif last < last_sma and sma_slope < 0:
        short_raw += W_TREND
    else:
        # trend net değilse düşük puan
        if last > last_sma:
            long_raw += W_TREND * 0.35
        elif last < last_sma:
            short_raw += W_TREND * 0.35

    # MACD hist (pozitif -> long, negatif -> short)
    # hist büyüklüğünü normalize et
    hist_norm = clamp(abs(hist) / (abs(last) * 0.002 + 1e-9), 0.0, 1.0)
    if hist > 0:
        long_raw += W_MACD * hist_norm
    elif hist < 0:
        short_raw += W_MACD * hist_norm

    # ADX (trend güç filtresi): adx düşükse ikisini de kırp
    # 18 altı zayıf, 25 üstü iyi
    adx_mult = 0.65 + 0.35 * clamp((adx_v - 18.0) / 12.0, 0.0, 1.0)
    # ayrıca ADX bileşen puanı da ekleyelim (trend varsa)
    adx_bonus = W_ADX * clamp((adx_v - 18.0) / 18.0, 0.0, 1.0)
    long_raw += adx_bonus * (1.0 if last > last_sma else 0.4)
    short_raw += adx_bonus * (1.0 if last < last_sma else 0.4)

    # ATR% (aşırı oynaksa ceza)
    # 1-4% ideal, 6% üstü risk -> puanı kır
    if atr_pct <= 4.0:
        atr_mult = 1.0
    elif atr_pct >= 8.0:
        atr_mult = 0.75
    else:
        # 4..8 lineer düş
        atr_mult = 1.0 - (atr_pct - 4.0) * (0.25 / 4.0)
    # ATR bileşeni: düşük-orta volatiliteye küçük bonus
    atr_bonus = W_ATR * clamp((4.0 - atr_pct) / 4.0, 0.0, 1.0)
    long_raw += atr_bonus * 0.6
    short_raw += atr_bonus * 0.6

    # Volume spike: trend yönüyle uyumluysa bonus
    vol_boost = W_VOL * clamp((vol_ratio - 1.2) / 1.0, 0.0, 1.0)
    if last > last_sma:
        long_raw += vol_boost
    elif last < last_sma:
        short_raw += vol_boost

    # -------------- multipliers --------------
    long_raw *= adx_mult * atr_mult
    short_raw *= adx_mult * atr_mult

    # -------------- final --------------
    # 0..100'e normalize et (ağırlık toplamına göre)
    weight_sum = float(W_RSI + W_BB + W_TREND + W_MACD + W_ADX + W_ATR + W_VOL)
    long_score = (long_raw / weight_sum) * 100.0
    short_score = (short_raw / weight_sum) * 100.0

    if long_score >= short_score:
        raw_best = int(round(clamp(long_score, 0, 100)))
        score_q = quantize_score(long_score, SCORE_STEP)
        direction = "LONG"
    else:
        raw_best = int(round(clamp(short_score, 0, 100)))
        score_q = quantize_score(short_score, SCORE_STEP)
        direction = "SHORT"

    return score_q, raw_best, direction


# =============================
# UI (Koyu tema fix + loader)
# =============================
st.set_page_config(page_title="KuCoin PRO Sniper — Auto (LONG+SHORT)", layout="wide")

st.markdown(
    """
<style>
/* Koyu tema sabit */
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
h1,h2,h3,h4,p,span,div { color: #e6edf3 !important; }
[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
[data-testid="stToolbar"] { visibility: hidden; height: 0px; }

/* Tablo yazıları daha net */
thead tr th { color: #e6edf3 !important; background: #0f172a !important; }
tbody tr td { color: #e6edf3 !important; background: #0b0f14 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# Auto refresh
def try_autorefresh(interval_ms: int, key: str):
    try:
        return st.autorefresh(interval=interval_ms, key=key)
    except Exception:
        try:
            return st.experimental_autorefresh(interval=interval_ms, key=key)
        except Exception:
            return None

try_autorefresh(interval_ms=int(AUTO_REFRESH_SEC * 1000), key="auto_refresh")


# Başlık
now_ist = datetime.now(IST)
st.title("KuCoin PRO Sniper — Auto (LONG + SHORT)")
st.caption(
    f"TF={TIMEFRAME} • STRONG: SKOR≥{STRONG_MIN_SCORE} • Tablo: önce STRONG, boş kalırsa TOP ile dolar • Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s"
)
st.markdown(
    f"""
<div style="text-align:right; margin-top:-40px;">
  <div style="font-size:12px; opacity:0.85;">Istanbul Time</div>
  <div style="font-size:18px; font-weight:800;">{now_ist.strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
""",
    unsafe_allow_html=True,
)


# =============================
# TARMA
# =============================
ex = make_exchange()

with st.spinner("⏳ KuCoin USDT spot evreni taranıyor..."):
    syms = load_usdt_spot_symbols()
    tickers = safe_fetch_tickers(ex)

# Likidite filtre + cap
ranked = []
for s in syms:
    t = tickers.get(s)
    qv = qv_24h(t)
    if qv >= MIN_QUOTE_VOL_24H:
        ranked.append((s, qv))

ranked.sort(key=lambda x: x[1], reverse=True)
scan_list = [s for s, _ in ranked[:MAX_SCAN_SYMBOLS]]

st.info(f"Evren (USDT spot): {len(syms)} • Likidite filtresi sonrası: {len(ranked)} • Tarama: {len(scan_list)}", icon="🧠")

progress = st.progress(0)
status = st.empty()

rows = []
total = max(1, len(scan_list))

for i, symbol in enumerate(scan_list, start=1):
    if i == 1:
        status.write("🔎 Mum verileri çekiliyor, skor hesaplanıyor...")

    progress.progress(int((i / total) * 100))

    try:
        ohlcv = safe_fetch_ohlcv(ex, symbol, TIMEFRAME, CANDLE_LIMIT)
        if not ohlcv or len(ohlcv) < 80:
            continue

        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        score_q, raw_best, direction = score_one(df)

        t = tickers.get(symbol, {})
        last = t.get("last", None)
        if last is None:
            last = float(df["close"].iloc[-1])
        else:
            try:
                last = float(last)
            except Exception:
                last = float(df["close"].iloc[-1])

        qv = qv_24h(t)

        rows.append(
            {
                "YÖN": direction,
                "COIN": symbol.replace("/USDT", ""),
                "SKOR": int(score_q),
                "FİYAT": float(last),
                "RAW": int(raw_best),
                "QV_24H": float(qv),
            }
        )

    except (ccxt.RequestTimeout, ccxt.NetworkError):
        continue
    except Exception:
        continue

    time.sleep(0.02)

progress.empty()
status.empty()

out = pd.DataFrame(rows)
if out.empty:
    st.error("Sonuç yok. (KuCoin ağ / rate-limit olabilir) Bir sonraki auto refresh'i bekle.")
    st.stop()

# STRONG öncelik, sonra TOP ile doldur
strong_df = out[out["SKOR"] >= STRONG_MIN_SCORE].copy()
rest_df = out[out["SKOR"] < STRONG_MIN_SCORE].copy()

strong_df = strong_df.sort_values(["SKOR", "QV_24H"], ascending=[False, False])
rest_df = rest_df.sort_values(["SKOR", "QV_24H"], ascending=[False, False])

final = pd.concat([strong_df, rest_df], axis=0).drop_duplicates(subset=["COIN"]).head(TABLE_SIZE).reset_index(drop=True)

if len(strong_df) > 0:
    st.success(f"✅ STRONG bulundu. Önce STRONG gösteriliyor (SKOR≥{STRONG_MIN_SCORE}), boş kalırsa TOP ile doluyor.", icon="✅")
else:
    st.warning(f"⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu.", icon="⚠️")

# Renklendirme (YÖN ve SKOR)
def style_table(df: pd.DataFrame):
    def dir_style(v):
        if str(v) == "LONG":
            return "background-color:#064e3b; color:#ffffff; font-weight:800;"
        if str(v) == "SHORT":
            return "background-color:#7f1d1d; color:#ffffff; font-weight:800;"
        return ""

    def score_style(v):
        try:
            v = int(v)
        except Exception:
            return ""
        if v >= STRONG_MIN_SCORE:
            return "background-color:#065f46; color:#ffffff; font-weight:900;"
        if v >= 75:
            return "background-color:#14532d; color:#ffffff; font-weight:800;"
        if v <= 35:
            return "background-color:#7f1d1d; color:#ffffff; font-weight:800;"
        return ""

    fmt = {
        "FİYAT": "{:.6f}",
        "QV_24H": "{:,.0f}",
    }

    return (
        df.style.format(fmt)
        .applymap(dir_style, subset=["YÖN"])
        .applymap(score_style, subset=["SKOR"])
        .set_properties(**{"border-color": "#1f2a37"})
    )

st.subheader("🎯 SNIPER TABLO")
st.dataframe(style_table(final), use_container_width=True, height=720)

st.caption(
    "Not: Bu sürüm ‘menüsüz’ ve otomatik. İstersen bir sonraki adımda MAX_SCAN_SYMBOLS/likidite filtresini ayarlanabilir yaparız (ama sen istemediğin için şimdilik sabit)."
)
