# app.py
# KuCoin PRO Sniper — Auto (LONG + SHORT)
# Baseline: koyu tema • skor adımı 5 • LONG/SHORT renklendirme • STRONG + TOP doldurma
# Not: pandas_ta KULLANMIYOR (requirements derdi çıkmasın)

import time
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timezone, timedelta

# ----------------------------
# SABİTLER (SIDEBAR YOK)
# ----------------------------
TF = "15min"
CANDLE_LIMIT = 200

MAX_SYMBOLS = 450              # pratik limit (KuCoin USDT evreninden hacme göre)
REFRESH_SEC = 240              # auto refresh
VOLVALUE_MIN = 20_000          # 24h quote volume altı elenir (illiquid azaltır)

# STRONG eşikleri (RAW üzerinden)
STRONG_LONG_RAW = 90
STRONG_SHORT_RAW = 10

# GATE seviyesi (Level-2): STRONG için 6/6 şart, fallback için >=2 yeter
STRONG_GATES_REQUIRED = 6
FALLBACK_MIN_GATES = 2

# Skor adımı
SCORE_STEP = 5

# ----------------------------
# STREAMLIT SAYFA AYARI
# ----------------------------
st.set_page_config(page_title="KuCoin PRO Sniper", layout="wide")

# Auto refresh (ekstra paket yok, JS ile)
components.html(
    f"""
    <script>
      setTimeout(function() {{
        window.location.reload();
      }}, {REFRESH_SEC*1000});
    </script>
    """,
    height=0,
)

# Koyu tema + okunurluk (şeffaflık/white sorunu için)
st.markdown(
    """
    <style>
      html, body, [class*="css"] {
        background-color: #0b0f14 !important;
        color: #e8eef6 !important;
      }
      .block-container { padding-top: 1.2rem; }
      h1, h2, h3 { color: #e8eef6 !important; }
      .sniper-sub { color: #9fb0c3; margin-top: -10px; }
      /* Kartlar */
      .sniper-card {
        background: #0f1722;
        border: 1px solid #1f2a3a;
        border-radius: 14px;
        padding: 14px 16px;
      }
      .sniper-warn {
        background: #2a2a12;
        border: 1px solid #444416;
        border-radius: 14px;
        padding: 14px 16px;
      }
      .sniper-ok {
        background: #0f2a1a;
        border: 1px solid #1f5a35;
        border-radius: 14px;
        padding: 14px 16px;
      }
      /* Dataframe */
      [data-testid="stDataFrame"] { background: #0b0f14 !important; }
      .stProgress > div > div > div > div { background-color: #2b7cff !important; }
      /* Gizli beyaz boşluklar */
      section.main > div { background-color: #0b0f14 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Başlık
st.markdown("## 🎯 KuCoin PRO Sniper — Auto (LONG + SHORT)")
st.markdown(
    f'<div class="sniper-sub">TF={TF} • STRONG: SKOR≥{STRONG_LONG_RAW} (LONG) / SKOR≤{STRONG_SHORT_RAW} (SHORT) • 6 Kapı • Skor adımı: {SCORE_STEP} • Auto: {REFRESH_SEC}s</div>',
    unsafe_allow_html=True
)

IST = timezone(timedelta(hours=3))
now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"<div class='sniper-sub'>İstanbul Time: {now_ist}</div>", unsafe_allow_html=True)

# ----------------------------
# YARDIMCI: SAYISAL İNDİKATÖRLER (pandas_ta yok)
# ----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    # Basit ADX (Wilder tarzı yaklaşıma yakın)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)

    atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / (atr_s.replace(0, np.nan))
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / (atr_s.replace(0, np.nan))

    dx = (100 * (plus_di - minus_di).abs() / ((plus_di + minus_di).replace(0, np.nan))).fillna(0)
    adx_v = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_v.fillna(0)

def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return (macd_line - signal_line).fillna(0)

def round_to_step(x: float, step: int) -> int:
    return int(round(x / step) * step)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# ----------------------------
# KUCOIN API
# ----------------------------
KUCOIN = "https://api.kucoin.com"

@st.cache_data(ttl=REFRESH_SEC, show_spinner=False)
def fetch_all_tickers():
    url = f"{KUCOIN}/api/v1/market/allTickers"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()["data"]["ticker"]
    df = pd.DataFrame(data)
    # beklenen kolonlar: symbolName, symbol, buy, sell, changeRate, vol, volValue, last
    return df

@st.cache_data(ttl=REFRESH_SEC, show_spinner=False)
def fetch_candles(symbol: str, tf: str = TF, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    # KuCoin candles: [time, open, close, high, low, volume, turnover]
    url = f"{KUCOIN}/api/v1/market/candles"
    params = {"type": tf, "symbol": symbol}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    arr = r.json()["data"]
    if not arr:
        return pd.DataFrame()
    arr = arr[:limit]
    df = pd.DataFrame(arr, columns=["time", "open", "close", "high", "low", "volume", "turnover"])
    # time unix seconds string
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    for c in ["open", "close", "high", "low", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("time")
    df = df.dropna()
    return df.reset_index(drop=True)

def is_usdt_spot(symbol_name: str) -> bool:
    return isinstance(symbol_name, str) and symbol_name.endswith("-USDT")

def is_junk(symbol_name: str) -> bool:
    # çok kaba filtre: leveraged / 3L 3S / UP DOWN vb.
    s = symbol_name.upper()
    bad = ["UP-", "DOWN-", "3L-", "3S-", "BULL-", "BEAR-"]
    return any(x in s for x in bad)

# ----------------------------
# BTC/ETH REJİM (basit ve stabil)
# ----------------------------
def regime_label() -> tuple[str, str]:
    """
    returns: (regime_text, bias) bias in {"LONG", "SHORT", "NEUTRAL", "NO_TRADE"}
    """
    try:
        btc = fetch_candles("BTC-USDT", TF, 200)
        eth = fetch_candles("ETH-USDT", TF, 200)
        if btc.empty or eth.empty:
            return ("REGIME: NO TRADE • BTC/ETH data yok", "NO_TRADE")

        def calc_bias(df):
            c = df["close"]
            e50 = ema(c, 50)
            e200 = ema(c, 200)
            r = rsi(c, 14)
            # trend + momentum
            long_ok = (c.iloc[-1] > e200.iloc[-1]) and (e50.iloc[-1] > e200.iloc[-1]) and (r.iloc[-1] >= 50)
            short_ok = (c.iloc[-1] < e200.iloc[-1]) and (e50.iloc[-1] < e200.iloc[-1]) and (r.iloc[-1] <= 50)
            if long_ok and not short_ok:
                return "LONG"
            if short_ok and not long_ok:
                return "SHORT"
            return "NEUTRAL"

        b1 = calc_bias(btc)
        b2 = calc_bias(eth)

        if b1 == "LONG" and b2 == "LONG":
            return ("REGIME: LONG BIAS • BTC/ETH bullish", "LONG")
        if b1 == "SHORT" and b2 == "SHORT":
            return ("REGIME: SHORT BIAS • BTC/ETH bearish", "SHORT")
        return ("REGIME: NEUTRAL • BTC/ETH neutral (bias yok)", "NEUTRAL")
    except Exception:
        return ("REGIME: NO TRADE • BTC/ETH error", "NO_TRADE")

# ----------------------------
# 6 KAPI + RAW SKOR
# RAW: 0..100 (0=çok güçlü SHORT, 100=çok güçlü LONG)
# ----------------------------
def compute_raw_and_gates(cdf: pd.DataFrame) -> tuple[float, int]:
    if cdf is None or cdf.empty or len(cdf) < 60:
        return (50.0, 0)

    close = cdf["close"]
    high = cdf["high"]
    low = cdf["low"]
    vol = cdf["volume"]

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e200 = ema(close, 200)

    r = rsi(close, 14)
    h = macd_hist(close)
    a = adx(high, low, close, 14)
    at = atr(high, low, close, 14)

    last_close = close.iloc[-1]
    last_rsi = r.iloc[-1]
    last_hist = h.iloc[-1]
    last_adx = a.iloc[-1]
    last_atr_pct = (at.iloc[-1] / last_close) if last_close else 0.0

    # ------------- 6 GATE (Level-2) -------------
    gates = []

    # Gate 1: Trend (EMA200 + EMA50 konumu)
    gate1_long = (last_close > e200.iloc[-1]) and (e50.iloc[-1] > e200.iloc[-1])
    gate1_short = (last_close < e200.iloc[-1]) and (e50.iloc[-1] < e200.iloc[-1])
    gates.append(gate1_long or gate1_short)

    # Gate 2: Momentum (RSI)
    gate2_long = last_rsi >= 55
    gate2_short = last_rsi <= 45
    gates.append(gate2_long or gate2_short)

    # Gate 3: MACD histogram yönü
    gate3_long = last_hist > 0
    gate3_short = last_hist < 0
    gates.append(gate3_long or gate3_short)

    # Gate 4: Trend strength (ADX)
    gates.append(last_adx >= 18)

    # Gate 5: Volume (son hacim / ort hacim)
    vmean = vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else vol.mean()
    gates.append(bool(vmean) and (vol.iloc[-1] >= (vmean * 1.2)))

    # Gate 6: Volatility/Noise filtresi (ATR%)
    # Çok düşük ATR = illiquid / fake; çok yüksek ATR = spike/noise
    gates.append(0.002 <= last_atr_pct <= 0.08)

    gate_pass = int(sum(bool(x) for x in gates))

    # ------------- RAW (0..100) -------------
    # Baz RAW: RSI + trend mesafesi + MACD + ADX (hepsi yumuşak)
    # Bu ham skor sonra 0..100'e sıkıştırılır.
    # LONG'u büyütür, SHORT'u küçültür.

    # RSI bileşeni (50 merkez)
    rsi_component = (last_rsi - 50) * 1.2  # -60..+60 civarı

    # Trend bileşeni (close/EMA200)
    trend_component = 0.0
    if e200.iloc[-1] != 0:
        trend_component = ((last_close / e200.iloc[-1]) - 1.0) * 400  # -? .. +?

    # MACD hist
    macd_component = np.tanh(last_hist * 5) * 18  # -18..+18

    # ADX (güç varsa LONG/SHORT sinyalini "keskinleştirir" ama yön vermez)
    adx_boost = np.tanh((last_adx - 18) / 10) * 10  # -10..+10

    # Kapı sayısı bonusu (çok yumuşak, yapıyı bozmasın)
    gate_bonus = (gate_pass - 3) * 3  # -9..+9

    raw_unclipped = 50 + rsi_component + trend_component + macd_component + adx_boost + gate_bonus

    # 0..100 clamp
    raw = float(np.clip(raw_unclipped, 0, 100))

    return (raw, gate_pass)

# ----------------------------
# STYLING (Base bozulmadan: LONG yeşil, SHORT kırmızı, STRONG daha koyu)
# ----------------------------
def style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    # Renkler
    LONG_BG = "#0f3b2e"          # normal long
    SHORT_BG = "#4a1212"         # normal short
    STRONG_LONG_BG = "#0a2a20"   # daha koyu long
    STRONG_SHORT_BG = "#2c0a0a"  # daha koyu short

    BASE_TEXT = "color: #e8eef6;"
    BORDER = "border-color: #1f2a3a !important;"
    HEAD_BG = "#101826"

    def row_style(row):
        direction = str(row.get("YÖN", ""))
        strong = bool(row.get("STRONG", False))

        if direction == "LONG":
            bg = STRONG_LONG_BG if strong else LONG_BG
        elif direction == "SHORT":
            bg = STRONG_SHORT_BG if strong else SHORT_BG
        else:
            bg = "#0f1722"

        return [f"background-color: {bg}; {BASE_TEXT} {BORDER}"] * len(row)

    sty = (
        df.style
        .apply(row_style, axis=1)
        .set_table_styles([
            {"selector": "th", "props": [("background-color", HEAD_BG), ("color", "#e8eef6"), ("border", "1px solid #1f2a3a")]},
            {"selector": "td", "props": [("border", "1px solid #1f2a3a")]},
            {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
        ])
        .format({
            "FİYAT": "{:.6f}",
            "QV_24H": "{:,.0f}",
            "RAW": "{:.0f}",
            "SKOR": "{:.0f}",
            "KAPI": "{:d}",
        }, na_rep="-")
    )
    return sty

# ----------------------------
# EVRENİ TOPLA + HESAPLA
# ----------------------------
def build_table():
    # progress/loader
    prog = st.progress(0, text="⏳ KuCoin USDT evreni hazırlanıyor...")
    t0 = time.time()

    tickers = fetch_all_tickers()

    # USDT spot + junk hariç
    tickers = tickers[tickers["symbolName"].apply(is_usdt_spot)]
    tickers = tickers[~tickers["symbolName"].apply(is_junk)]

    # Likidite filtresi (volValue)
    tickers["volValue"] = pd.to_numeric(tickers.get("volValue", 0), errors="coerce").fillna(0.0)
    tickers = tickers[tickers["volValue"] >= VOLVALUE_MIN]

    # Hacme göre sırala ve pratik limit uygula
    tickers = tickers.sort_values("volValue", ascending=False).head(MAX_SYMBOLS).copy()

    universe_count = int(len(tickers))
    prog.progress(10, text=f"🧠 Evren (USDT spot): {universe_count} • Likidite filtresi aktif • Tarama başlıyor...")

    # Regime
    reg_txt, bias = regime_label()
    prog.progress(15, text=f"📈 Rejim okunuyor... ({bias})")

    # Tarama
    rows = []
    scanned = 0
    total = universe_count if universe_count else 1

    for _, r in tickers.iterrows():
        sym = r["symbolName"]  # örn: TRX-USDT
        qv = safe_float(r.get("volValue", 0.0), 0.0)
        last = safe_float(r.get("last", 0.0), 0.0)

        try:
            cdf = fetch_candles(sym, TF, CANDLE_LIMIT)
            raw, gate_pass = compute_raw_and_gates(cdf)

            # bias etkisi (çok hafif, base mantığı bozmasın)
            # SHORT bias: raw'ı biraz aşağı iter, LONG bias: yukarı iter
            if bias == "SHORT":
                raw = float(np.clip(raw - 2.0, 0, 100))
            elif bias == "LONG":
                raw = float(np.clip(raw + 2.0, 0, 100))

            skor = round_to_step(raw, SCORE_STEP)
            yon = "LONG" if raw >= 50 else "SHORT"

            # STRONG kriteri: hem eşik hem de 6/6 kapı
            strong = False
            if gate_pass >= STRONG_GATES_REQUIRED:
                if yon == "LONG" and raw >= STRONG_LONG_RAW:
                    strong = True
                if yon == "SHORT" and raw <= STRONG_SHORT_RAW:
                    strong = True

            rows.append({
                "YÖN": yon,
                "COIN": sym.replace("-USDT", ""),
                "SKOR": int(skor),
                "FİYAT": float(last) if last else float(cdf["close"].iloc[-1]) if (cdf is not None and not cdf.empty) else float("nan"),
                "RAW": float(raw),
                "QV_24H": float(qv),
                "KAPI": int(gate_pass),
                "STRONG": bool(strong),
            })

        except Exception:
            # tek coin fail olursa atla
            pass

        scanned += 1
        if scanned % 15 == 0 or scanned == total:
            p = 15 + int((scanned / total) * 80)
            prog.progress(min(p, 95), text=f"🔎 Taranıyor... {scanned}/{total}")

    prog.progress(96, text="🧩 STRONG + TOP doldurma hazırlanıyor...")

    df = pd.DataFrame(rows)
    if df.empty:
        prog.progress(100, text="❌ Veri gelmedi (network/KuCoin).")
        return reg_txt, bias, universe_count, 0, pd.DataFrame()

    # ----------------------------
    # STRONG seç (base mantık): önce STRONG, kalan boşlukları TOP ile doldur
    # ----------------------------
    # STRONG havuzu
    strong_df = df[df["STRONG"] == True].copy()

    # LONG tarafında büyük RAW iyidir; SHORT tarafında küçük RAW iyidir.
    # STRONG'ları önce en "uç" olandan başlayarak sırala.
    strong_df["_rank"] = strong_df.apply(
        lambda x: (100 - x["RAW"]) if x["YÖN"] == "LONG" else x["RAW"],
        axis=1
    )
    strong_df = strong_df.sort_values(["_rank"], ascending=True).drop(columns=["_rank"])

    # hedef tablo boyutu
    TARGET_N = 20

    # ---- fallback TOP havuzu (AŞIRI SERT OLMASIN) ----
    # Burada kritik düzeltme:
    # STRONG yoksa bile tablo dolsun diye fallback'te 6/6 kapıyı ZORUNLU yapmıyoruz.
    # Sadece en az 2 kapı + likidite zaten filtreli.
    fallback_pool = df[df["KAPI"] >= FALLBACK_MIN_GATES].copy()

    # "en yakın" sıralaması:
    # LONG: RAW 90'a yakın
    # SHORT: RAW 10'a yakın
    def proximity(row):
        if row["YÖN"] == "LONG":
            return abs(row["RAW"] - STRONG_LONG_RAW)
        return abs(row["RAW"] - STRONG_SHORT_RAW)

    fallback_pool["_prox"] = fallback_pool.apply(proximity, axis=1)
    # aynı prox'ta daha yüksek kapı daha iyi
    fallback_pool = fallback_pool.sort_values(["_prox", "KAPI", "QV_24H"], ascending=[True, False, False])

    # ---- 10/10 denge (estetik) ----
    # Önce STRONG'ları al, sonra eksik kalan tarafı TOP ile tamamla.
    result = strong_df.copy()

    def add_rows(from_df, need, exclude_set):
        out = []
        for _, rr in from_df.iterrows():
            key = rr["COIN"]
            if key in exclude_set:
                continue
            out.append(rr)
            exclude_set.add(key)
            if len(out) >= need:
                break
        return out

    used = set(result["COIN"].tolist())
    # hedef: 10 long + 10 short (ama STRONG çoksa taşabilir, yine de TARGET_N'e kadar doldur)
    want_long = 10
    want_short = 10

    cur_long = int((result["YÖN"] == "LONG").sum()) if not result.empty else 0
    cur_short = int((result["YÖN"] == "SHORT").sum()) if not result.empty else 0

    need_long = max(0, want_long - cur_long)
    need_short = max(0, want_short - cur_short)

    fb_long = fallback_pool[fallback_pool["YÖN"] == "LONG"]
    fb_short = fallback_pool[fallback_pool["YÖN"] == "SHORT"]

    addL = add_rows(fb_long, need_long, used)
    addS = add_rows(fb_short, need_short, used)

    if addL:
        result = pd.concat([result, pd.DataFrame(addL)], ignore_index=True)
    if addS:
        result = pd.concat([result, pd.DataFrame(addS)], ignore_index=True)

    # hâlâ eksik varsa (bazı günler SHORT ya da LONG çok az olabilir),
    # kalan boşluğu prox sırasına göre (karma) doldur.
    if len(result) < TARGET_N:
        need = TARGET_N - len(result)
        addAny = add_rows(fallback_pool, need, used)
        if addAny:
            result = pd.concat([result, pd.DataFrame(addAny)], ignore_index=True)

    # yine de hedefe ulaşamadıysa (çok nadir), en azından en iyi prox adaylarıyla doldur
    result = result.head(TARGET_N).copy()

    # FINAL tablo: SKOR'a göre (LONG yüksek, SHORT düşük) üstte görünsün diye
    # Kullanıcı "top" istiyor: LONG için yüksek skor; SHORT için düşük skor daha güçlü.
    # Karma sıralama: STRONG önce, sonra prox, sonra likidite
    result["_prox"] = result.apply(proximity, axis=1)
    result = result.sort_values(
        ["STRONG", "_prox", "QV_24H"],
        ascending=[False, True, False]
    ).drop(columns=["_prox"])

    prog.progress(100, text=f"✅ Bitti • {int(time.time()-t0)}s")

    return reg_txt, bias, universe_count, len(df), result.reset_index(drop=True)

# ----------------------------
# ÇALIŞTIR + CACHE (boş ekranda son tabloyu göster)
# ----------------------------
if "last_df" not in st.session_state:
    st.session_state.last_df = pd.DataFrame()
if "last_msg" not in st.session_state:
    st.session_state.last_msg = ""

try:
    reg_txt, bias, ucount, scanned_count, out = build_table()

    # üst rejim bandı
    if "SHORT BIAS" in reg_txt:
        st.markdown(f"<div class='sniper-warn'>⛔ {reg_txt}</div>", unsafe_allow_html=True)
    elif "LONG BIAS" in reg_txt:
        st.markdown(f"<div class='sniper-ok'>✅ {reg_txt}</div>", unsafe_allow_html=True)
    elif "NEUTRAL" in reg_txt:
        st.markdown(f"<div class='sniper-warn'>⚠️ {reg_txt}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='sniper-warn'>⚠️ {reg_txt}</div>", unsafe_allow_html=True)

    # evren kartı
    if out is None or out.empty:
        # bu durumda son başarılı tabloyu göster
        if st.session_state.last_df is not None and not st.session_state.last_df.empty:
            st.markdown(
                "<div class='sniper-warn'>⚠️ Bu turda veri gelmedi (network/KuCoin). Son başarılı tarama gösteriliyor.</div>",
                unsafe_allow_html=True
            )
            out = st.session_state.last_df.copy()
        else:
            st.markdown(
                "<div class='sniper-warn'>⚠️ Aday yok (network/KuCoin veya filtre çok sert). Bir sonraki auto refresh’i bekle.</div>",
                unsafe_allow_html=True
            )

    # out doluysa cachele
    if out is not None and not out.empty:
        st.session_state.last_df = out.copy()

    # ----------------------------
    # SAYILAR (Strong long/short + long/short)
    # ----------------------------
    if out is not None and not out.empty:
        strong_long = int(((out["YÖN"] == "LONG") & (out["STRONG"] == True)).sum())
        strong_short = int(((out["YÖN"] == "SHORT") & (out["STRONG"] == True)).sum())
        long_cnt = int((out["YÖN"] == "LONG").sum())
        short_cnt = int((out["YÖN"] == "SHORT").sum())

        st.markdown(
            f"""
            <div class="sniper-card">
              🧠 Evren (USDT spot): <b>{ucount}</b> • Tarama: <b>{scanned_count}</b><br>
              ✅ <b>STRONG LONG:</b> {strong_long} • 🟥 <b>STRONG SHORT:</b> {strong_short} •
              🟩 <b>LONG:</b> {long_cnt} • 🟥 <b>SHORT:</b> {short_cnt}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## 🎯 SNIPER TABLO")
    if out is not None and not out.empty:
        show = out[["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI", "STRONG"]].copy()

        # STRONG metni (görselde ayrı renk zaten var ama checkbox gibi dursun)
        # istersen STRONG kolonu gizleyebiliriz; şimdilik dursun.
        sty = style_table(show)

        st.dataframe(sty, use_container_width=True, hide_index=True)
    else:
        st.markdown("<div class='sniper-warn'>⚠️ Tablo boş.</div>", unsafe_allow_html=True)

except Exception as e:
    # Tam çöküşte de son tabloyu bas
    if st.session_state.last_df is not None and not st.session_state.last_df.empty:
        st.markdown(
            "<div class='sniper-warn'>⚠️ Uygulama bu turda hata verdi. Son başarılı tarama gösteriliyor.</div>",
            unsafe_allow_html=True
        )
        show = st.session_state.last_df.copy()
        show = show[["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI", "STRONG"]]
        st.dataframe(style_table(show), use_container_width=True, hide_index=True)
    else:
        st.error(f"Hata: {e}")
