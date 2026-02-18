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

AUTO_REFRESH_SEC = 240          # sayfa kaç sn'de bir yenilensin
TABLE_ROWS = 20                 # tabloda kaç satır görünsün
STRONG_SCORE = 90               # STRONG eşiği
MIN_QV_USDT_24H = 0             # 0 bırak: "tüm coinler" (istersen 5000 gibi filtre koyarsın)
MAX_SYMBOLS = None              # None = tüm USDT spot evreni (yavaş olabilir)

# Skor adımı: 5'er 5'er
SCORE_STEP = 5

# İndikatör parametreleri
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
EMA_FAST = 12
EMA_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
VOL_Z_PERIOD = 20
VWAP_PERIOD = 20


# =============================
# YARDIMCI MATH
# =============================
def _clip(x, lo=0.0, hi=1.0):
    return float(np.clip(x, lo, hi))


def _sigmoid(x: float) -> float:
    # stabil sigmoid
    x = float(np.clip(x, -20, 20))
    return 1.0 / (1.0 + np.exp(-x))


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi_wilder(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def bollinger(series: pd.Series, period: int, n_std: float):
    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def vwap_typical(df: pd.DataFrame, period: int) -> pd.Series:
    # rolling VWAP (typical price)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    pv = tp * vol
    pv_sum = pv.rolling(period, min_periods=period).sum()
    vol_sum = vol.rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    return (pv_sum / vol_sum).fillna(method="bfill").fillna(method="ffill")


def quantize_score(x: float, step: int = 5) -> int:
    x = float(np.clip(x, 0, 100))
    return int(round(x / step) * step)


# =============================
# KUCOIN / CCXT
# =============================
def make_exchange() -> ccxt.kucoin:
    return ccxt.kucoin({"enableRateLimit": True, "timeout": 20000})


@st.cache_data(show_spinner=False, ttl=600)
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
    syms = sorted(set(syms))
    return syms


def safe_fetch_tickers(ex: ccxt.Exchange) -> dict:
    try:
        return ex.fetch_tickers()
    except Exception:
        return {}


def qv_usdt_24h(t: dict) -> float:
    # KuCoin tickers genelde quoteVolume verir
    if not isinstance(t, dict):
        return 0.0
    qv = t.get("quoteVolume")
    if qv is not None:
        try:
            return float(qv)
        except Exception:
            pass
    # fallback: baseVolume * last
    try:
        bv = float(t.get("baseVolume", 0.0))
        last = float(t.get("last", 0.0))
        return bv * last
    except Exception:
        return 0.0


def spread_pct(t: dict) -> float:
    # bid/ask varsa spread cezası için kullan
    try:
        bid = float(t.get("bid", 0.0))
        ask = float(t.get("ask", 0.0))
        last = float(t.get("last", 0.0))
        if bid > 0 and ask > 0 and last > 0:
            return (ask - bid) / last
    except Exception:
        pass
    return 0.0


def safe_fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int):
    return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


# =============================
# SKORLAMA (Daha “sert” + daha gerçekçi)
# =============================
def score_one(df: pd.DataFrame, ticker: dict) -> tuple[str, int, int]:
    """
    returns: (direction, score, raw_score)
      - direction: LONG / SHORT
      - score: quantized score (0..100 step=5)
      - raw_score: before quantize (0..100 int)
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    # indicators
    rsi14 = rsi_wilder(close, RSI_PERIOD)
    _, bb_up, bb_low = bollinger(close, BB_PERIOD, BB_STD)
    atr14 = atr(high, low, close, ATR_PERIOD)

    ema_f = ema(close, EMA_FAST)
    ema_s = ema(close, EMA_SLOW)

    macd_line = ema_f - ema_s
    macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False, min_periods=MACD_SIGNAL).mean()
    macd_hist = macd_line - macd_signal

    vwap20 = vwap_typical(df, VWAP_PERIOD)

    # last values
    last = float(close.iloc[-1])
    last_atr = float(atr14.iloc[-1]) if not np.isnan(atr14.iloc[-1]) else 0.0
    last_rsi = float(rsi14.iloc[-1])
    last_bb_up = float(bb_up.iloc[-1])
    last_bb_low = float(bb_low.iloc[-1])
    last_vwap = float(vwap20.iloc[-1])

    if last_atr <= 0 or np.isnan([last_bb_up, last_bb_low, last_vwap]).any():
        return "—", 0, 0

    # bb position (0 low .. 1 high)
    denom = max(1e-12, (last_bb_up - last_bb_low))
    bbp = (last - last_bb_low) / denom
    bbp = float(np.clip(bbp, 0.0, 1.0))

    # trend & momentum normalized by ATR
    trend_norm = float((ema_f.iloc[-1] - ema_s.iloc[-1]) / last_atr)
    macd_norm = float(macd_hist.iloc[-1] / last_atr)

    # volume z-score
    v_mean = vol.rolling(VOL_Z_PERIOD, min_periods=VOL_Z_PERIOD).mean().iloc[-1]
    v_std = vol.rolling(VOL_Z_PERIOD, min_periods=VOL_Z_PERIOD).std(ddof=0).iloc[-1]
    vol_z = 0.0
    if pd.notna(v_mean) and pd.notna(v_std) and float(v_std) > 0:
        vol_z = float((vol.iloc[-1] - v_mean) / v_std)

    # vwap deviation
    vwap_dev = (last - last_vwap) / max(1e-12, last_vwap)

    # --- LONG components (0..1)
    c_trend_L = _sigmoid(trend_norm * 1.2)                 # trend +
    c_macd_L = _sigmoid(macd_norm * 1.4)                   # momentum +
    # RSI: çok yüksek değil, orta + hafif oversold daha iyi
    c_rsi_L = float(np.exp(-((last_rsi - 52.0) / 12.0) ** 2))
    # pullback: alt banda yakınsa iyi (bbp küçük)
    c_bb_L = 1.0 - bbp
    c_bb_L = _clip(c_bb_L, 0, 1)
    # VWAP: vwap altında/çevresinde iyi
    c_vwap_L = _sigmoid((-vwap_dev) / 0.010)
    # volume confirm
    c_vol = _sigmoid((vol_z - 0.2) * 1.0)

    long_raw = (
        0.22 * c_trend_L
        + 0.18 * c_macd_L
        + 0.18 * c_rsi_L
        + 0.18 * c_bb_L
        + 0.14 * c_vwap_L
        + 0.10 * c_vol
    ) * 100.0

    # --- SHORT components (0..1)
    c_trend_S = _sigmoid((-trend_norm) * 1.2)              # trend -
    c_macd_S = _sigmoid((-macd_norm) * 1.4)                # momentum -
    c_rsi_S = float(np.exp(-((last_rsi - 48.0) / 12.0) ** 2))
    # bounce: üst banda yakınsa iyi (bbp büyük)
    c_bb_S = bbp
    c_bb_S = _clip(c_bb_S, 0, 1)
    # VWAP: vwap üstünde/çevresinde iyi
    c_vwap_S = _sigmoid((vwap_dev) / 0.010)

    short_raw = (
        0.22 * c_trend_S
        + 0.18 * c_macd_S
        + 0.18 * c_rsi_S
        + 0.18 * c_bb_S
        + 0.14 * c_vwap_S
        + 0.10 * c_vol
    ) * 100.0

    # direction
    if long_raw >= short_raw:
        direction = "LONG"
        raw = float(long_raw)
    else:
        direction = "SHORT"
        raw = float(short_raw)

    # --- Liquidity / Spread penalties (100'leri azaltan ana şey)
    qv = qv_usdt_24h(ticker)
    sp = spread_pct(ticker)

    # likidite faktörü: 0..1 (çok küçük hacimde skor kırılır)
    # 1e4=10k -> düşük, 1e6=1M -> iyi, 1e7=10M -> çok iyi
    liq = _clip(np.log10(qv + 1.0) / 7.0, 0.0, 1.0)  # 10^7 ~ 1.0
    liq_factor = 0.55 + 0.45 * liq                   # min 0.55

    # spread cezası: %0.5 spread bile kaldıraçta kötü
    # sp=0.001 (0.1%) -> hafif kır, sp=0.005 (0.5%) -> çok kır
    spread_factor = _clip(1.0 - (sp * 120.0), 0.40, 1.0)

    raw = raw * liq_factor * spread_factor

    # hard cap + quantize
    raw = float(np.clip(raw, 0, 100))
    score = quantize_score(raw, SCORE_STEP)

    return direction, score, int(round(raw))


# =============================
# TABLO SEÇİMİ (STRONG + TOP fill)
# =============================
def build_table(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all

    df_all = df_all.sort_values(["SKOR", "QV_24H"], ascending=[False, False]).reset_index(drop=True)

    strong = df_all[df_all["SKOR"] >= STRONG_SCORE].copy()
    strong = strong.sort_values(["SKOR", "QV_24H"], ascending=[False, False]).reset_index(drop=True)

    if len(strong) >= TABLE_ROWS:
        return strong.head(TABLE_ROWS).reset_index(drop=True)

    picked = strong.copy()
    remain = df_all[~df_all["COIN"].isin(picked["COIN"])].copy()
    remain = remain.sort_values(["SKOR", "QV_24H"], ascending=[False, False]).reset_index(drop=True)

    need = TABLE_ROWS - len(picked)
    if need > 0 and not remain.empty:
        picked = pd.concat([picked, remain.head(need)], ignore_index=True)

    # SHORT görünmeme problemini engelle: imkan varsa min 3 SHORT ve min 3 LONG
    total_shorts = int((df_all["YÖN"] == "SHORT").sum())
    total_longs = int((df_all["YÖN"] == "LONG").sum())

    def enforce_min(direction: str, min_n: int):
        nonlocal picked
        cur = int((picked["YÖN"] == direction).sum())
        if cur >= min_n:
            return
        if direction == "SHORT" and total_shorts < min_n:
            return
        if direction == "LONG" and total_longs < min_n:
            return

        # eksik kadar en iyi adayları al
        missing = min_n - cur
        candidates = df_all[(df_all["YÖN"] == direction) & (~df_all["COIN"].isin(picked["COIN"]))].copy()
        if candidates.empty:
            return
        add = candidates.sort_values(["SKOR", "QV_24H"], ascending=[False, False]).head(missing).copy()

        # karşı yönden en düşük skorları çıkarıp yerine koy
        opp = "LONG" if direction == "SHORT" else "SHORT"
        drop_idx = picked[picked["YÖN"] == opp].sort_values(["SKOR", "QV_24H"], ascending=[True, True]).head(len(add)).index
        picked = picked.drop(index=drop_idx).reset_index(drop=True)
        picked = pd.concat([picked, add], ignore_index=True)

    enforce_min("SHORT", 3)
    enforce_min("LONG", 3)

    picked = picked.sort_values(["SKOR", "QV_24H"], ascending=[False, False]).head(TABLE_ROWS).reset_index(drop=True)
    return picked


# =============================
# UI (Dark + Auto)
# =============================
st.set_page_config(page_title="KuCoin PRO Sniper — Auto (LONG + SHORT)", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 2.0rem; }
h1,h2,h3,h4,h5,h6,p,span,div { color: #e6edf3; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stToolbar"] { display: none; }

.small-muted { opacity: 0.85; font-size: 13px; }
.badge { padding: 10px 12px; border-radius: 10px; border: 1px solid #1f2a37; }
.badge-ok { background: rgba(16, 185, 129, 0.12); }
.badge-warn { background: rgba(245, 158, 11, 0.12); }
.badge-bad { background: rgba(239, 68, 68, 0.12); }

</style>
""",
    unsafe_allow_html=True,
)

# Auto refresh (tıklama yok)
try:
    st.autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
except Exception:
    try:
        st.experimental_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
    except Exception:
        pass

left, right = st.columns([2, 1], vertical_alignment="center")
with left:
    st.title("🎯 KuCoin PRO Sniper — Auto (LONG + SHORT)")
    st.caption(
        f"TF={TIMEFRAME} • STRONG: SKOR≥{STRONG_SCORE} • Tablo: önce STRONG, sonra TOP ile dolar • Skor adımı: {SCORE_STEP}"
    )
with right:
    now_ist = datetime.now(IST)
    st.markdown(
        f"""
<div style="text-align:right; padding-top: 10px;">
  <div class="small-muted">Istanbul Time</div>
  <div style="font-size: 18px; font-weight: 800;">{now_ist.strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
""",
        unsafe_allow_html=True,
    )

status_box = st.empty()
progress_box = st.empty()

# ================
# RUN SCAN
# ================
ex = make_exchange()

with st.spinner("⏳ KuCoin USDT spot evreni taranıyor… (lütfen bekle)"):
    syms = load_usdt_spot_symbols()
    tickers = safe_fetch_tickers(ex)

    # "tüm coinler" = tüm USDT spot pair; ama istersen MIN_QV ile çöp coinleri azaltırsın
    candidates = []
    for s in syms:
        t = tickers.get(s, {})
        qv = qv_usdt_24h(t)
        if qv < MIN_QV_USDT_24H:
            continue
        candidates.append(s)

    if MAX_SYMBOLS is not None:
        candidates = candidates[: int(MAX_SYMBOLS)]

    total = len(candidates)
    if total == 0:
        status_box.error("Hiç USDT spot sembol bulunamadı (veya filtre çok sert).")
        st.stop()

    progress = progress_box.progress(0, text=f"Başlıyor… (0/{total})")
    rows = []
    last_ok = 0

    for i, symbol in enumerate(candidates, start=1):
        if i == 1 or i % 15 == 0:
            progress.progress(int((i - 1) / total * 100), text=f"Taranıyor: {symbol}  ({i}/{total})")

        try:
            ohlcv = safe_fetch_ohlcv(ex, symbol, TIMEFRAME, CANDLE_LIMIT)
            if not ohlcv or len(ohlcv) < max(EMA_SLOW, BB_PERIOD, RSI_PERIOD, ATR_PERIOD, VOL_Z_PERIOD, VWAP_PERIOD) + 5:
                continue

            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            t = tickers.get(symbol, {})

            direction, score, raw = score_one(df, t)
            if direction == "—":
                continue

            last_price = float(df["close"].iloc[-1])

            rows.append(
                {
                    "YÖN": direction,
                    "COIN": symbol.replace("/USDT", ""),
                    "SKOR": int(score),
                    "FİYAT": float(last_price),
                    "RAW": int(raw),
                    "QV_24H": int(round(qv_usdt_24h(t))),
                }
            )
            last_ok += 1

        except (ccxt.RequestTimeout, ccxt.NetworkError):
            pass
        except ccxt.ExchangeError:
            pass
        except Exception:
            pass

        time.sleep(0.03)  # rate limit dostu

    progress.progress(100, text=f"Tamamlandı. Başarılı: {last_ok}/{total}")
    time.sleep(0.2)
    progress_box.empty()

df_all = pd.DataFrame(rows)
if df_all.empty:
    status_box.markdown(
        f"""<div class="badge badge-bad">❌ Veri gelmedi. KuCoin/network anlık sorun olabilir. Bir sonraki auto-refresh'te tekrar dene.</div>""",
        unsafe_allow_html=True,
    )
    st.stop()

df_table = build_table(df_all)

strong_count = int((df_table["SKOR"] >= STRONG_SCORE).sum())
longs = int((df_table["YÖN"] == "LONG").sum())
shorts = int((df_table["YÖN"] == "SHORT").sum())

if strong_count > 0:
    status_box.markdown(
        f"""<div class="badge badge-ok">✅ STRONG bulundu. STRONG sayısı: <b>{strong_count}</b> • LONG/SHORT: <b>{longs}/{shorts}</b></div>""",
        unsafe_allow_html=True,
    )
else:
    status_box.markdown(
        f"""<div class="badge badge-warn">⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu • LONG/SHORT: <b>{longs}/{shorts}</b></div>""",
        unsafe_allow_html=True,
    )

st.write("")
st.subheader("🎯 SNIPER TABLO")

# =============================
# STYLING (Dark table)
# =============================
def style_df(df: pd.DataFrame):
    d = df.copy()

    fmt = {
        "FİYAT": "{:.6f}",
        "QV_24H": "{:,}",
        "SKOR": "{:d}",
        "RAW": "{:d}",
    }

    def dir_style(v):
        if v == "LONG":
            return "background-color:#064e3b;color:#e6edf3;font-weight:800;"
        if v == "SHORT":
            return "background-color:#7f1d1d;color:#e6edf3;font-weight:800;"
        return ""

    def score_style(v):
        try:
            v = int(v)
        except Exception:
            return ""
        if v >= STRONG_SCORE:
            return "background-color:#065f46;color:#ffffff;font-weight:900;"
        if v >= 80:
            return "background-color:#0f766e;color:#ffffff;font-weight:800;"
        if v >= 70:
            return "background-color:#1f2937;color:#e6edf3;"
        return "background-color:#111827;color:#e6edf3;"

    return (
        d.style.format(fmt)
        .applymap(dir_style, subset=["YÖN"])
        .applymap(score_style, subset=["SKOR"])
        .set_table_styles(
            [
                {"selector": "th", "props": [("background-color", "#0f172a"), ("color", "#e6edf3"), ("border-color", "#1f2a37")]},
                {"selector": "td", "props": [("background-color", "#0b0f14"), ("color", "#e6edf3"), ("border-color", "#1f2a37")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("border", "1px solid #1f2a37")]},
            ]
        )
    )

# gösterilecek kolon sırası
show_cols = ["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H"]
df_show = df_table.loc[:, show_cols].copy()

st.dataframe(style_df(df_show), height=750)
