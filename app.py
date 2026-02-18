import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================================================
# BASELINE AYARLAR (SIDEBAR YOK - kullanıcı ayarı yok)
# =========================================================
TF = "15min"
CANDLE_LIMIT = 200
AUTO_REFRESH_SEC = 240

# STRONG eşikleri (SENİN İSTEDİĞİN)
STRONG_LONG_RAW = 90
STRONG_SHORT_RAW = 10

# Skor step (5'lik adım)
SCORE_STEP = 5

# Tablo kalabalık olmasın (10/10 estetik)
N_LONG = 10
N_SHORT = 10
N_TOTAL = N_LONG + N_SHORT

# Likidite / spread filtreleri (çok sert değil, ama çöpleri eleyelim)
MIN_QV_24H = 50_000        # quote volume (USDT) altını ele
MAX_SPREAD_PCT = 0.40      # %0.40 üstü spread ele (illiquid)

# Tarama üst limiti (performans için)
MAX_SCAN = 450

# =========================================================
# STREAMLIT PAGE
# =========================================================
st.set_page_config(page_title="KuCoin PRO Sniper", layout="wide")

# =========================================================
# DARK THEME CSS (ŞEFFAFLIK / OKUNMA PROBLEMİNİ ÇÖZER)
# =========================================================
st.markdown(
    """
    <style>
      html, body, [class*="css"]  { background: #0b0f14 !important; }
      .block-container { padding-top: 1.2rem; }
      h1, h2, h3, h4, h5, h6, p, span, div { color: #e7edf5 !important; }
      .stAlert > div { background: #0f1722 !important; color: #e7edf5 !important; border-radius: 14px !important; }
      .sniper-card {
        background:#0f1722; border:1px solid rgba(255,255,255,.08);
        padding:14px 16px; border-radius:14px; margin-bottom:10px;
      }
      .muted { color:#a7b3c3 !important; font-size: 0.95rem; }
      .tiny { color:#9aa6b7 !important; font-size: 0.85rem; }
      .pill-ok { background:#11351f; border:1px solid rgba(0,255,120,.25); padding:10px 12px; border-radius:14px; }
      .pill-warn { background:#2a2a10; border:1px solid rgba(255,220,0,.25); padding:10px 12px; border-radius:14px; }
      .pill-bad { background:#331316; border:1px solid rgba(255,80,80,.25); padding:10px 12px; border-radius:14px; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# KUCOIN ENDPOINTS
# =========================================================
KUCOIN = "https://api.kucoin.com"


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def now_istanbul_str():
    # İstanbul UTC+3 (sabit)
    utc_now = datetime.now(timezone.utc)
    ist = utc_now.astimezone(timezone.utc).replace(hour=(utc_now.hour + 3) % 24)
    # Basit gösterim (Streamlit için yeterli)
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


@st.cache_data(ttl=60)
def get_all_tickers():
    r = requests.get(f"{KUCOIN}/api/v1/market/allTickers", timeout=15)
    r.raise_for_status()
    data = r.json()["data"]["ticker"]
    df = pd.DataFrame(data)

    # Symbol format: "TRX-USDT"
    df = df[df["symbol"].str.endswith("-USDT")].copy()

    # numeric
    df["buy"] = df["buy"].map(_safe_float)
    df["sell"] = df["sell"].map(_safe_float)
    df["last"] = df["last"].map(_safe_float)
    df["volValue"] = df["volValue"].map(_safe_float)  # quote volume
    df["changeRate"] = df["changeRate"].map(_safe_float)

    # spread %
    df["spread_pct"] = np.where(
        (df["buy"] > 0) & (df["sell"] > 0),
        (df["sell"] - df["buy"]) / df["sell"] * 100.0,
        np.nan
    )

    # bazı “garip” token isimleri (UP/DOWN/3L/3S vb.) istersen ekleriz; şimdilik dokunmuyoruz
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["last", "volValue"])

    # Likidite + spread filtresi
    df_f = df[(df["volValue"] >= MIN_QV_24H) & (df["spread_pct"] <= MAX_SPREAD_PCT)].copy()

    # En likitleri tarayalım
    df_f = df_f.sort_values("volValue", ascending=False).head(MAX_SCAN).reset_index(drop=True)
    return df, df_f


@st.cache_data(ttl=120)
def get_candles(symbol: str, tf: str = TF, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    # KuCoin candles: /api/v1/market/candles?type=15min&symbol=TRX-USDT
    params = {"type": tf, "symbol": symbol}
    r = requests.get(f"{KUCOIN}/api/v1/market/candles", params=params, timeout=15)
    r.raise_for_status()
    rows = r.json()["data"]

    # rows: [time, open, close, high, low, volume, turnover]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["ts", "open", "close", "high", "low", "volume", "turnover"])
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="s")
    for c in ["open", "close", "high", "low", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("ts").reset_index(drop=True)
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df


# =========================================================
# INDICATORS (pandas_ta yok - requirements basit)
# =========================================================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    return m - s


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


# =========================================================
# REGIME (BTC/ETH)
# =========================================================
def calc_regime() -> tuple[str, float]:
    """
    returns (regime_label, bias_score)
    bias_score: +1 bullish, -1 bearish, 0 neutral
    """
    try:
        btc = get_candles("BTC-USDT", TF, CANDLE_LIMIT)
        eth = get_candles("ETH-USDT", TF, CANDLE_LIMIT)
        if btc.empty or eth.empty:
            return ("NO TRADE • BTC/ETH unknown", 0.0)

        def bias_one(x: pd.DataFrame) -> int:
            c = x["close"]
            e20 = ema(c, 20).iloc[-1]
            e50 = ema(c, 50).iloc[-1]
            r = rsi(c, 14).iloc[-1]
            # trend + momentum
            score = 0
            score += 1 if e20 > e50 else -1
            score += 1 if r > 52 else (-1 if r < 48 else 0)
            return score

        b_btc = bias_one(btc)
        b_eth = bias_one(eth)

        total = b_btc + b_eth  # -4..+4
        if total >= 2:
            return ("LONG BIAS • BTC/ETH bullish", +1.0)
        if total <= -2:
            return ("SHORT BIAS • BTC/ETH bearish", -1.0)
        return ("NEUTRAL • BTC/ETH neutral (bias yok)", 0.0)
    except Exception:
        return ("NO TRADE • BTC/ETH error", 0.0)


# =========================================================
# 6 KAPI + RAW(0-100) + SKOR(5'lik)
# =========================================================
def compute_signal(symbol: str, qv_24h: float, last_price: float, regime_bias: float) -> dict | None:
    df = get_candles(symbol, TF, CANDLE_LIMIT)
    if df.empty or len(df) < 60:
        return None

    c = df["close"]
    e20 = ema(c, 20)
    e50 = ema(c, 50)
    r = rsi(c, 14)
    h = macd_hist(c)
    a = atr(df, 14)

    close = float(c.iloc[-1])
    ema20 = float(e20.iloc[-1])
    ema50 = float(e50.iloc[-1])
    rsi14 = float(r.iloc[-1])
    hist = float(h.iloc[-1])
    atr14 = float(a.iloc[-1])

    # --- KAPI değerlendirmeleri (bullishness tarafı) ---
    # Gate1: Trend (EMA20 > EMA50)
    g1 = 1 if ema20 > ema50 else -1

    # Gate2: RSI momentum
    if rsi14 > 55:
        g2 = 1
    elif rsi14 < 45:
        g2 = -1
    else:
        g2 = 0

    # Gate3: MACD hist
    if hist > 0:
        g3 = 1
    elif hist < 0:
        g3 = -1
    else:
        g3 = 0

    # Gate4: 15m "impulse" (son kapanış EMA20’den ne kadar uzak)
    # ATR bazlı: close > ema20 + 0.35*ATR => bullish, close < ema20 - 0.35*ATR => bearish
    if atr14 > 0:
        if close > ema20 + 0.35 * atr14:
            g4 = 1
        elif close < ema20 - 0.35 * atr14:
            g4 = -1
        else:
            g4 = 0
    else:
        g4 = 0

    # Gate5: Volatility kalite (ATR% yeterli mi?) => sinyal sertliği için
    # ATR% çok düşükse (ör. <0.25) “sert sinyal” üretmeyelim: nötrleştirir.
    atr_pct = (atr14 / close * 100.0) if close > 0 else 0
    if atr_pct >= 0.60:
        g5 = 1  # “tradeable volatility”
    elif atr_pct <= 0.25:
        g5 = -1  # “ölü piyasa”
    else:
        g5 = 0

    # Gate6: Regime (BTC/ETH)
    # +1 bullish, -1 bearish, 0 neutral
    g6 = 1 if regime_bias > 0 else (-1 if regime_bias < 0 else 0)

    gates = [g1, g2, g3, g4, g5, g6]

    # --- RAW (bullishness 0..100) ---
    # Temel: 50. Her gate +/- “puan” ekler.
    # Sertliği artırmak için (6 kapı) gate ağırlığını biraz yükselttik.
    raw = 50.0

    # Gate ağırlıkları (toplam etkisi ~ +/-45)
    weights = [12, 10, 10, 8, 5, 6]  # toplam 51 ama 0/± ile dengelenir
    for gi, wi in zip(gates, weights):
        raw += gi * wi

    # RSI ince ayar (50 üstü bullish, 50 altı bearish)
    raw += (rsi14 - 50.0) * 0.35  # ~ +/-17.5 max

    # MACD hist ince ayar (normalize basitçe ATR ile)
    if atr14 > 0:
        raw += np.clip(hist / atr14, -2, 2) * 4.0  # +/-8

    raw = float(np.clip(raw, 0, 100))

    # SKOR 5'lik adım
    skor = int(round(raw / SCORE_STEP) * SCORE_STEP)
    skor = int(np.clip(skor, 0, 100))

    # Direction: bullishness >=50 => LONG, else SHORT
    direction = "LONG" if raw >= 50 else "SHORT"

    # Kapı sayısı (seçilen yön için kaç kapı "uyumlu")
    # LONG: gate = +1, SHORT: gate = -1
    if direction == "LONG":
        kapi = sum(1 for g in gates if g == 1)
    else:
        kapi = sum(1 for g in gates if g == -1)

    strong = (direction == "LONG" and raw >= STRONG_LONG_RAW and kapi >= 6) or \
             (direction == "SHORT" and raw <= STRONG_SHORT_RAW and kapi >= 6)

    return {
        "YÖN": direction,
        "COIN": symbol.replace("-USDT", ""),
        "SKOR": skor,
        "FİYAT": float(last_price) if last_price else close,
        "RAW": int(round(raw)),
        "QV_24H": float(qv_24h),
        "KAPI": int(kapi),
        "STRONG": bool(strong),
    }


# =========================================================
# TABLO SEÇİMİ (STRONG önce, kalan boşluk TOP ile dolar)
# =========================================================
def pick_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df

    strong_df = df[df["STRONG"]].copy()

    # STRONG sıralama: LONG yüksek RAW, SHORT düşük RAW
    strong_long = strong_df[strong_df["YÖN"] == "LONG"].sort_values(["RAW", "QV_24H"], ascending=[False, False])
    strong_short = strong_df[strong_df["YÖN"] == "SHORT"].sort_values(["RAW", "QV_24H"], ascending=[True, False])

    strong_ordered = pd.concat([strong_long, strong_short], ignore_index=True)

    # Kalanlar: TOP adaylar (uçlara en yakınlar)
    # LONG için RAW desc, SHORT için RAW asc
    rest = df[~df["STRONG"]].copy()
    rest_long = rest[rest["YÖN"] == "LONG"].sort_values(["RAW", "QV_24H"], ascending=[False, False])
    rest_short = rest[rest["YÖN"] == "SHORT"].sort_values(["RAW", "QV_24H"], ascending=[True, False])

    # Denge: 10 LONG + 10 SHORT
    out_rows = []
    used = set()

    def take_rows(xdf, n):
        nonlocal out_rows, used
        for _, r in xdf.iterrows():
            if len(out_rows) >= N_TOTAL:
                break
            key = (r["COIN"], r["YÖN"])
            if key in used:
                continue
            out_rows.append(r)
            used.add(key)
            if sum(1 for rr in out_rows if rr["YÖN"] == xdf.iloc[0]["YÖN"]) >= n:
                break

    # STRONG'ları önce al (dengeyi mümkünse koru)
    # STRONG içinde zaten karışık gelebilir; count bazlı toplayalım
    out_long = 0
    out_short = 0
    for _, r in strong_ordered.iterrows():
        if len(out_rows) >= N_TOTAL:
            break
        if r["YÖN"] == "LONG" and out_long >= N_LONG:
            continue
        if r["YÖN"] == "SHORT" and out_short >= N_SHORT:
            continue
        key = (r["COIN"], r["YÖN"])
        if key in used:
            continue
        out_rows.append(r)
        used.add(key)
        out_long += 1 if r["YÖN"] == "LONG" else 0
        out_short += 1 if r["YÖN"] == "SHORT" else 0

    # Kalanı TOP ile doldur
    # Önce eksik yönü tamamla
    need_long = max(0, N_LONG - out_long)
    need_short = max(0, N_SHORT - out_short)

    if need_long > 0 and not rest_long.empty:
        for _, r in rest_long.iterrows():
            if need_long <= 0:
                break
            key = (r["COIN"], r["YÖN"])
            if key in used:
                continue
            out_rows.append(r)
            used.add(key)
            need_long -= 1

    if need_short > 0 and not rest_short.empty:
        for _, r in rest_short.iterrows():
            if need_short <= 0:
                break
            key = (r["COIN"], r["YÖN"])
            if key in used:
                continue
            out_rows.append(r)
            used.add(key)
            need_short -= 1

    # Hâlâ boşluk varsa (çok sert filtre / network), kalan en iyi uçlarla doldur (denge bozulabilir)
    if len(out_rows) < N_TOTAL:
        rest_all = pd.concat([rest_long, rest_short], ignore_index=True)
        # uçlara yakınlık: max(raw, 100-raw)
        rest_all["EXTREME"] = rest_all["RAW"].apply(lambda x: max(x, 100 - x))
        rest_all = rest_all.sort_values(["EXTREME", "QV_24H"], ascending=[False, False])
        for _, r in rest_all.iterrows():
            if len(out_rows) >= N_TOTAL:
                break
            key = (r["COIN"], r["YÖN"])
            if key in used:
                continue
            out_rows.append(r)
            used.add(key)

    out_df = pd.DataFrame(out_rows).copy()
    if not out_df.empty:
        # Son görünüm için kolon sırası
        out_df = out_df[["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI", "STRONG"]]
        # STRONG önce gelsin, sonra TOP (SKOR’a göre)
        out_df["_rank"] = out_df.apply(
            lambda r: (0 if r["STRONG"] else 1,
                       -r["RAW"] if r["YÖN"] == "LONG" else r["RAW"],
                       -r["QV_24H"]),
            axis=1
        )
        out_df = out_df.sort_values("_rank").drop(columns=["_rank"]).reset_index(drop=True)

    return strong_df, out_df


# =========================================================
# TABLE STYLING (LONG yeşil / SHORT kırmızı; STRONG daha koyu)
# =========================================================
def style_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    if df.empty:
        return df.style

    # renkler
    LONG_BG = "#0f3d2e"          # normal long
    SHORT_BG = "#4a1518"         # normal short
    STRONG_LONG_BG = "#0a2a1f"   # daha koyu long
    STRONG_SHORT_BG = "#2e0c0f"  # daha koyu short

    TEXT = "color: #e7edf5; font-weight: 600;"
    BORDER = "border: 1px solid rgba(255,255,255,.07);"

    def row_style(row):
        is_strong = bool(row.get("STRONG", False))
        if row["YÖN"] == "LONG":
            bg = STRONG_LONG_BG if is_strong else LONG_BG
        else:
            bg = STRONG_SHORT_BG if is_strong else SHORT_BG
        return [f"background-color: {bg}; {TEXT} {BORDER}"] * len(row)

    sty = df.style.apply(row_style, axis=1)

    # header + genel
    sty = sty.set_table_styles(
        [
            {"selector": "th", "props": [("background-color", "#0f1722"),
                                         ("color", "#e7edf5"),
                                         ("border", "1px solid rgba(255,255,255,.08)"),
                                         ("font-weight", "700")]},
            {"selector": "td", "props": [("border", "1px solid rgba(255,255,255,.06)")]},
        ]
    )

    # format
    sty = sty.format(
        {
            "FİYAT": "{:.6f}",
            "QV_24H": "{:,.0f}",
        }
    )
    return sty


# =========================================================
# UI
# =========================================================
st.title("🎯 KuCoin PRO Sniper — Auto (LONG + SHORT)")
st.markdown(
    f"<div class='muted'>TF={TF} • STRONG: SKOR≥{STRONG_LONG_RAW} (LONG) / SKOR≤{STRONG_SHORT_RAW} (SHORT) • 6 Kapı • Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s</div>",
    unsafe_allow_html=True
)
st.markdown(f"<div class='tiny'>İstanbul Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

# LOADING / PROGRESS alanı
progress = st.progress(0, text="⏳ KuCoin USDT spot evreni taranıyor...")

# REGIME
regime_label, regime_bias = calc_regime()
if "LONG BIAS" in regime_label:
    st.markdown(f"<div class='pill-ok'>🟢 REGIME: {regime_label}</div>", unsafe_allow_html=True)
elif "SHORT BIAS" in regime_label:
    st.markdown(f"<div class='pill-bad'>🔴 REGIME: {regime_label}</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='pill-warn'>⚠️ REGIME: {regime_label}</div>", unsafe_allow_html=True)

# TICKERS
all_df, universe = get_all_tickers()

evren_total = len(all_df[all_df["symbol"].str.endswith("-USDT")])
evren_after = len(universe)

# Tarama sayısı = evren_after (MAX_SCAN'e göre zaten kırpılıyor)
st.markdown(
    f"<div class='sniper-card'>🧠 Evren (USDT spot): <b>{evren_total}</b> • Likidite filtresi sonrası: <b>{evren_after}</b> • Tarama: <b>{evren_after}</b></div>",
    unsafe_allow_html=True
)

# TARAMA
rows = []
start = time.time()

with st.spinner("🔎 Sinyaller hesaplanıyor (RAW/Skor/6 Kapı)..."):
    for i, r in universe.iterrows():
        symbol = r["symbol"]
        qv = float(r["volValue"])
        last = float(r["last"])

        out = compute_signal(symbol, qv, last, regime_bias)
        if out:
            rows.append(out)

        # progress
        if evren_after > 0:
            pct = int((i + 1) / evren_after * 100)
            progress.progress(min(pct, 100), text=f"⏳ Tarama: {i+1}/{evren_after} • {pct}%")

progress.progress(100, text=f"✅ Bitti • {int(time.time()-start)}s")

df = pd.DataFrame(rows)

# Eğer tarama sonucu boş gelirse fallback mesaj
if df.empty:
    st.markdown(
        "<div class='pill-warn'>⚠️ Aday yok (network/KuCoin veya filtre çok sert). Bir sonraki auto refresh’i bekle.</div>",
        unsafe_allow_html=True
    )
    st.stop()

# STRONG + TOP seçimi
strong_df, table_df = pick_table(df)

# SAYIMLAR (SENİN İSTEDİĞİN: strong long/short ve total long/short)
strong_long_cnt = int(((df["STRONG"]) & (df["YÖN"] == "LONG")).sum())
strong_short_cnt = int(((df["STRONG"]) & (df["YÖN"] == "SHORT")).sum())
long_cnt = int((table_df["YÖN"] == "LONG").sum()) if not table_df.empty else 0
short_cnt = int((table_df["YÖN"] == "SHORT").sum()) if not table_df.empty else 0

# Renk ikonları: LONG yeşil, SHORT kırmızı (doğru)
st.markdown(
    f"""
    <div class='sniper-card'>
      ✅ <b>STRONG LONG</b>: {strong_long_cnt} 🟩 &nbsp;&nbsp;
      ✅ <b>STRONG SHORT</b>: {strong_short_cnt} 🟥<br/>
      <b>LONG</b>: {long_cnt} 🟩 &nbsp;&nbsp;
      <b>SHORT</b>: {short_cnt} 🟥
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("🎯 SNIPER TABLO")

# STRONG mesajı
if (strong_long_cnt + strong_short_cnt) > 0:
    st.markdown(
        "<div class='pill-ok'>✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu.</div>",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div class='pill-warn'>⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu.</div>",
        unsafe_allow_html=True
    )

# tablo
sty = style_table(table_df)
st.dataframe(sty, use_container_width=True, height=520)

# AUTO REFRESH
time.sleep(0.1)
st.markdown(f"<div class='tiny'>Auto refresh: {AUTO_REFRESH_SEC}s</div>", unsafe_allow_html=True)
st.write("")  # spacing
st.write("")  # spacing

# Streamlit'te otomatik refresh için: st_autorefresh yoksa basit yöntem
# (Cloud'da genelde sorun çıkarmadan çalışır)
st.markdown(
    f"""
    <script>
      setTimeout(function() {{
        window.location.reload();
      }}, {AUTO_REFRESH_SEC * 1000});
    </script>
    """,
    unsafe_allow_html=True
)
