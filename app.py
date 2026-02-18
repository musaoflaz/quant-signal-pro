import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import streamlit as st


# =========================
# BASE AYARLAR (DEĞİŞTİRME)
# =========================
APP_TITLE = "KuCoin PRO Sniper — Auto (LONG + SHORT)"
TF = "15m"
AUTO_REFRESH_SEC = 240

# STRONG eşiği (BASE)
STRONG_LONG_MIN = 90   # SKOR >= 90 ve YÖN=LONG
STRONG_SHORT_MAX = 10  # SKOR <= 10 ve YÖN=SHORT

# Tablo kapasitesi (BASE)
TABLE_N = 20
FALLBACK_LONG_N = 10
FALLBACK_SHORT_N = 10

# Skor adımı (BASE)
SCORE_STEP = 5

# Tarama limiti (performans)
SCAN_LIMIT = 450  # önce en likit 450 (base ekranda "Tarama: 450" gibi)

# Likidite filtresi (USDT quote vol)
MIN_QV_24H = 10_000  # çok sert değil; base gibi çok coin kalsın

# KuCoin API
KUCOIN = "https://api.kucoin.com"


# =========================
# UI / TEMA (Koyu + okunaklı)
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown(
    """
    <style>
      /* Genel koyu zemin */
      .stApp {
        background: #0b0f14;
        color: #e8eef7;
      }

      /* Başlıkların şeffaf görünmesini engelle */
      h1,h2,h3, p, div, span {
        color: #e8eef7 !important;
      }

      /* Kartlar */
      .block-container { padding-top: 1.2rem; }

      /* Streamlit uyarı kutuları okunaklı */
      .stAlert > div { color: #e8eef7 !important; }

      /* Dataframe başlık satırı */
      thead tr th {
        background-color: #121a24 !important;
        color: #e8eef7 !important;
        border-bottom: 1px solid #223047 !important;
      }

      /* Dataframe satır çizgileri */
      tbody tr td {
        border-bottom: 1px solid #192435 !important;
      }

      /* Sağ üstteki ikon barı görünür kalsın */
      .stDataFrame div[role="toolbar"] { opacity: 0.9; }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
def ist_now_str():
    ist = timezone(timedelta(hours=3))
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")


def round_to_step(x, step=5):
    return int(np.round(x / step) * step)


def clamp(x, lo=0, hi=100):
    return float(max(lo, min(hi, x)))


def safe_get(url, params=None, timeout=12):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=AUTO_REFRESH_SEC, show_spinner=False)
def fetch_usdt_universe():
    """USDT spot evreni + 24h quote volume (volValue)"""
    j = safe_get(f"{KUCOIN}/api/v1/market/allTickers")
    if not j or "data" not in j or "ticker" not in j["data"]:
        return pd.DataFrame()

    tick = pd.DataFrame(j["data"]["ticker"])

    # Sadece USDT pariteleri
    tick = tick[tick["symbol"].str.endswith("-USDT", na=False)].copy()

    # Çöp/leveraged token filtreleri (basit)
    bad_suffix = ("-UP-USDT", "-DOWN-USDT", "3L-USDT", "3S-USDT", "5L-USDT", "5S-USDT")
    tick = tick[~tick["symbol"].str.endswith(bad_suffix, na=False)]

    # Quote volume (volValue) numeric
    if "volValue" in tick.columns:
        tick["QV_24H"] = pd.to_numeric(tick["volValue"], errors="coerce").fillna(0.0)
    elif "quoteVolume" in tick.columns:
        tick["QV_24H"] = pd.to_numeric(tick["quoteVolume"], errors="coerce").fillna(0.0)
    else:
        tick["QV_24H"] = 0.0

    # Last price
    tick["last"] = pd.to_numeric(tick.get("last"), errors="coerce")

    # Likidite filtresi
    tick = tick[tick["QV_24H"] >= MIN_QV_24H].copy()

    # En likitlere göre sırala
    tick = tick.sort_values("QV_24H", ascending=False)

    return tick[["symbol", "last", "QV_24H"]].head(SCAN_LIMIT).reset_index(drop=True)


@st.cache_data(ttl=AUTO_REFRESH_SEC, show_spinner=False)
def fetch_klines(symbol: str, tf: str = "15m", limit: int = 200):
    """KuCoin candlestick: [time, open, close, high, low, volume, turnover]"""
    # KuCoin: type param (1min,3min,5min,15min,1hour...)
    tf_map = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "2h": "2hour",
        "4h": "4hour",
        "1d": "1day",
    }
    ktype = tf_map.get(tf, "15min")

    j = safe_get(
        f"{KUCOIN}/api/v1/market/candles",
        params={"symbol": symbol, "type": ktype},
        timeout=12,
    )
    if not j or "data" not in j:
        return None

    data = j["data"]
    if not data:
        return None

    # KuCoin reverse chronological geliyor; çevirelim
    df = pd.DataFrame(
        data,
        columns=["ts", "open", "close", "high", "low", "volume", "turnover"],
    )
    for c in ["open", "close", "high", "low", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="s", utc=True)
    df = df.sort_values("ts").tail(limit).reset_index(drop=True)

    return df


def compute_raw_and_gates(df: pd.DataFrame):
    """
    RAW: 0-100 arası "long olasılığı" gibi davranır.
    - RAW>=50 => LONG
    - RAW<50  => SHORT
    STRONG:
    - LONG: RAW/SKOR >= 90
    - SHORT: RAW/SKOR <= 10
    6 kapı: "seviye 2" mantığında orta-sert filtre.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # İndikatörler
    rsi = ta.rsi(close, length=14)
    ema20 = ta.ema(close, length=20)
    ema50 = ta.ema(close, length=50)
    ema200 = ta.ema(close, length=200)
    bb = ta.bbands(close, length=20, std=2.0)
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    adx = ta.adx(high, low, close, length=14)
    stochrsi = ta.stochrsi(close, length=14)

    # son değerler
    rsi_v = float(rsi.iloc[-1]) if rsi is not None else 50.0
    ema20_v = float(ema20.iloc[-1]) if ema20 is not None else float(close.iloc[-1])
    ema50_v = float(ema50.iloc[-1]) if ema50 is not None else float(close.iloc[-1])
    ema200_v = float(ema200.iloc[-1]) if ema200 is not None else float(close.iloc[-1])

    bb_l = float(bb["BBL_20_2.0"].iloc[-1]) if bb is not None else float(close.iloc[-1])
    bb_u = float(bb["BBU_20_2.0"].iloc[-1]) if bb is not None else float(close.iloc[-1])
    bb_m = float(bb["BBM_20_2.0"].iloc[-1]) if bb is not None else float(close.iloc[-1])

    macdh = float(macd["MACDh_12_26_9"].iloc[-1]) if macd is not None else 0.0
    adx_v = float(adx["ADX_14"].iloc[-1]) if adx is not None else 0.0

    k = float(stochrsi["STOCHRSIk_14_14_3_3"].iloc[-1]) if stochrsi is not None else 50.0

    px = float(close.iloc[-1])

    # -----------------
    # 6 KAPI (Seviye 2)
    # -----------------
    gates = []

    # Kapı-1: Trend yönü (EMA20 vs EMA50)
    g1 = ema20_v > ema50_v
    gates.append(g1)

    # Kapı-2: Büyük trend filtresi (EMA200)
    g2 = px > ema200_v
    gates.append(g2)

    # Kapı-3: Momentum (MACD hist)
    g3 = macdh > 0
    gates.append(g3)

    # Kapı-4: RSI bandı (aşırı uç yok)
    g4 = (rsi_v >= 45) and (rsi_v <= 65)
    gates.append(g4)

    # Kapı-5: BB merkezine göre konum (mean-reversion tuzaklarını azaltır)
    g5 = px >= bb_m
    gates.append(g5)

    # Kapı-6: Trend gücü (ADX)
    g6 = adx_v >= 18
    gates.append(g6)

    gate_pass = int(sum(gates))  # 0..6

    # -----------------
    # RAW SKOR (0..100)
    # (Base davranış: Long yüksek, Short düşük)
    # -----------------
    raw = 50.0

    # Trend katkısı
    raw += 12 if g1 else -12
    raw += 8 if g2 else -8

    # Momentum
    raw += 10 if g3 else -10

    # RSI (seviye 2: aşırıya kaçmasın)
    if rsi_v > 60:
        raw += 6
    elif rsi_v < 40:
        raw -= 6

    # BB konumu
    # üst banda yakınsa long olasılığı düşer, alt banda yakınsa artar
    if bb_u > bb_l:
        pos = (px - bb_l) / (bb_u - bb_l)  # 0 alt band, 1 üst band
        raw += (0.5 - pos) * 18  # merkezden uzaklaştıkça etkisi artsın

    # Stoch RSI (aşırı alım/satım)
    if k < 20:
        raw += 6
    elif k > 80:
        raw -= 6

    # Kapı sayısı ile sertleştir
    # 6/6 ise uçlara yaklaşsın, düşükse merkeze çekilsin
    raw += (gate_pass - 3) * 3.5

    raw = clamp(raw, 0, 100)

    return raw, gate_pass


def build_table(universe: pd.DataFrame, progress=None, status=None):
    rows = []
    total = len(universe)

    for i, r in universe.iterrows():
        symbol = r["symbol"]
        last = float(r["last"]) if pd.notna(r["last"]) else np.nan
        qv = float(r["QV_24H"])

        if status:
            status.text(f"⏳ KuCoin USDT spot taranıyor... {i+1}/{total}  |  {symbol}")
        if progress:
            progress.progress((i + 1) / max(1, total))

        kl = fetch_klines(symbol, tf=TF, limit=200)
        if kl is None or len(kl) < 120:
            continue

        raw, gate_pass = compute_raw_and_gates(kl)

        direction = "LONG" if raw >= 50 else "SHORT"
        score = round_to_step(raw, SCORE_STEP)

        rows.append(
            {
                "YÖN": direction,
                "COIN": symbol.replace("-USDT", ""),
                "SKOR": int(score),
                "FİYAT": float(last) if np.isfinite(last) else float(kl["close"].iloc[-1]),
                "RAW": int(round(raw)),
                "QV_24H": int(round(qv)),
                "KAPI": int(gate_pass),
            }
        )

        # hafif throttle (rate limit riskini azaltır)
        time.sleep(0.01)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # STRONG tanımı (BASE)
    df["IS_STRONG"] = False
    df.loc[(df["YÖN"] == "LONG") & (df["SKOR"] >= STRONG_LONG_MIN), "IS_STRONG"] = True
    df.loc[(df["YÖN"] == "SHORT") & (df["SKOR"] <= STRONG_SHORT_MAX), "IS_STRONG"] = True

    # "en yakın aday" mantığı (BASE)
    def dist_to_strong(row):
        if row["YÖN"] == "LONG":
            return abs(row["RAW"] - STRONG_LONG_MIN)
        else:
            return abs(row["RAW"] - STRONG_SHORT_MAX)

    df["DIST"] = df.apply(dist_to_strong, axis=1)

    # Önce STRONG'lar + sonra TOP doldurma (BASE)
    strong = df[df["IS_STRONG"]].copy()

    # LONG: güçlüye yakın olanlar önce (RAW yüksek), SHORT: güçlüye yakın olanlar önce (RAW düşük)
    longs = df[df["YÖN"] == "LONG"].copy().sort_values(["IS_STRONG", "RAW", "QV_24H"], ascending=[False, False, False])
    shorts = df[df["YÖN"] == "SHORT"].copy().sort_values(["IS_STRONG", "RAW", "QV_24H"], ascending=[False, True, False])

    # STRONG zaten eklenecek; tekrarları çıkaracağız
    picked = []
    picked_set = set()

    def add_rows(src_df, need):
        nonlocal picked, picked_set
        for _, rr in src_df.iterrows():
            key = rr["COIN"]
            if key in picked_set:
                continue
            picked.append(rr)
            picked_set.add(key)
            if len(picked) >= need:
                break

    # 1) Tüm STRONG'lar (ama max TABLE_N’e kadar)
    if not strong.empty:
        strong_sorted = strong.sort_values(["YÖN", "DIST", "QV_24H"], ascending=[True, True, False])
        add_rows(strong_sorted, TABLE_N)

    # 2) Boşluk varsa 10/10 denge ile doldur (BASE)
    if len(picked) < TABLE_N:
        # mevcut long/short say
        cur_long = sum(1 for x in picked if x["YÖN"] == "LONG")
        cur_short = sum(1 for x in picked if x["YÖN"] == "SHORT")

        target_long = min(FALLBACK_LONG_N, TABLE_N)
        target_short = min(FALLBACK_SHORT_N, TABLE_N)

        need_long = max(0, target_long - cur_long)
        need_short = max(0, target_short - cur_short)

        add_rows(longs, min(TABLE_N, len(picked) + need_long))
        add_rows(shorts, min(TABLE_N, len(picked) + need_short))

    # 3) Hâlâ boşsa en iyi TOP adaylarla tamamla (BASE)
    if len(picked) < TABLE_N:
        # "en yakın" + likidite
        top = df.sort_values(["DIST", "QV_24H"], ascending=[True, False]).copy()
        add_rows(top, TABLE_N)

    out = pd.DataFrame(picked).copy()
    if out.empty:
        return out

    # Görsel düzen
    out = out.drop(columns=["DIST"], errors="ignore")
    out = out[["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI", "IS_STRONG"]].copy()

    return out


def style_table(df: pd.DataFrame):
    # LONG yeşil / SHORT kırmızı, STRONG daha koyu (BASE)
    def row_style(row):
        is_strong = bool(row.get("IS_STRONG", False))
        direction = str(row.get("YÖN", ""))

        if direction == "LONG":
            bg = "#0b3d2e" if not is_strong else "#046d3a"  # strong daha koyu
        elif direction == "SHORT":
            bg = "#5a0b0b" if not is_strong else "#8a0000"
        else:
            bg = "#101826"

        return [f"background-color: {bg}; color: #ffffff; font-weight: 600;"] * len(row)

    sty = (
        df.style
        .apply(row_style, axis=1)
        .format({
            "FİYAT": "{:,.6f}",
            "QV_24H": "{:,.0f}",
        })
    )
    return sty


def dataframe_render(styler_or_df, height=560):
    # Streamlit sürüm uyumu: width="stretch" varsa kullan, yoksa use_container_width
    try:
        st.dataframe(styler_or_df, width="stretch", height=height)
    except TypeError:
        st.dataframe(styler_or_df, use_container_width=True, height=height)


# =========================
# HEADER
# =========================
st.markdown(f"## 🎯 {APP_TITLE}")
st.caption(f"TF={TF} • STRONG: SKOR≥{STRONG_LONG_MIN} (LONG) / SKOR≤{STRONG_SHORT_MAX} (SHORT) • 6 Kapı • Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s")
st.caption(f"İstanbul Time: **{ist_now_str()}**")

# Auto refresh
st.markdown(
    f"""
    <script>
      const refreshMs = {AUTO_REFRESH_SEC * 1000};
      setTimeout(() => {{
        window.location.reload();
      }}, refreshMs);
    </script>
    """,
    unsafe_allow_html=True,
)


# =========================
# LOADING / PROGRESS
# =========================
status = st.empty()
progress = st.progress(0)

status.text("⏳ KuCoin USDT spot evreni çekiliyor...")
universe = fetch_usdt_universe()

if universe.empty:
    progress.empty()
    status.empty()
    st.error("KuCoin verisi alınamadı (network/limit). Biraz sonra tekrar dene.")
    st.stop()

# Evren bilgisi kutusu (BASE görünüm)
evren_n = int(len(universe))
info = st.info(f"🧠 Evren (USDT spot): {evren_n} • Likidite filtresi sonrası: {evren_n} • Tarama: {min(SCAN_LIMIT, evren_n)}")

status.text("⏳ KuCoin USDT spot taranıyor... (yükleniyor)")
table_df = build_table(universe, progress=progress, status=status)

progress.empty()
status.empty()

# =========================
# SONUÇ + SAYIM (İSTEDİĞİN)
# =========================
st.markdown("## 🎯 SNIPER TABLO")

if table_df.empty:
    st.warning("Aday yok (network/KuCoin veya filtre çok sert). Bir sonraki auto refresh’i bekle.")
    st.stop()

# Sayım (BASE bozmadan)
dfc = table_df.copy()
strong_long_n = int(((dfc["YÖN"] == "LONG") & (dfc["SKOR"] >= STRONG_LONG_MIN)).sum())
strong_short_n = int(((dfc["YÖN"] == "SHORT") & (dfc["SKOR"] <= STRONG_SHORT_MAX)).sum())
long_n = int((dfc["YÖN"] == "LONG").sum())
short_n = int((dfc["YÖN"] == "SHORT").sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("STRONG LONG", strong_long_n)
c2.metric("STRONG SHORT", strong_short_n)
c3.metric("LONG", long_n)
c4.metric("SHORT", short_n)

# Üst mesaj (BASE)
has_any_strong = (strong_long_n + strong_short_n) > 0
if has_any_strong:
    st.success("✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu.")
else:
    st.warning("⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu.")

# Tablo (renkler + strong koyu)
show_df = table_df.drop(columns=["IS_STRONG"], errors="ignore").copy()
styled = style_table(table_df.drop(columns=[], errors="ignore"))

# Görünüm: IS_STRONG kolonunu göstermiyoruz ama style için table_df üzerinden boyuyoruz.
# Bu yüzden render'da styler kullanıyoruz.
dataframe_render(styled, height=620)

st.caption("Not: RAW 0–100 arası 'LONG tarafı' eğilimi gibi düşün. RAW≥50 LONG, RAW<50 SHORT. SKOR = RAW'ın 5’lik adım ile yuvarlanmış hali (base).")
