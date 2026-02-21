import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import ccxt


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Sniper — Auto (LONG + SHORT)", layout="wide")

TF = "15m"
HTF = "1h"

TOP_N_PER_EXCHANGE = 180          # her borsadan hacimli ilk N (filtre sonrası)
TABLE_N_LONG = 10                 # UI tabloda LONG kaç satır
TABLE_N_SHORT = 10                # UI tabloda SHORT kaç satır
AUTO_REFRESH_SEC = 240            # Streamlit sayfa yenileme
REPORT_EVERY_MIN = 20             # STRONG yoksa rapor kaç dakikada bir Telegram

# 8 Kapı: STRONG için 8/8
GATES_TOTAL = 8
GATES_REQUIRED_STRONG = 8

# Skor adımı (2'şer)
SCORE_STEP = 2

# STRONG eşikleri (çok seçici)
# LONG: 8/8 -> 100 skor. 7/8 -> 88 olur (8'e bölünce) => strong olmasın.
STRONG_LONG_SCORE_MIN = 98        # pratikte 100
# SHORT: 8/8 -> 0 skor (SHORT skor ters) => strong short 0/2
STRONG_SHORT_SCORE_MAX = 2

# Likidite filtresi (çöp azaltma)
MIN_QV_24H_USDT = 300_000
MAX_SPREAD_PCT = 0.35

# ccxt rate limit
CCXT_RATE_LIMIT = True

# Telegram max list
TG_MAX_LIST = 20


# =========================
# TELEGRAM (SECRETS)
# =========================
def _get_secret(key: str):
    try:
        return st.secrets.get(key, None)
    except Exception:
        return None

TG_TOKEN = _get_secret("TG_TOKEN")
TG_CHAT_ID = _get_secret("TG_CHAT_ID")

def tg_enabled() -> bool:
    return bool(TG_TOKEN) and bool(TG_CHAT_ID)

def send_telegram_html(html_text: str) -> bool:
    if not tg_enabled():
        return False
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": html_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, data=payload, timeout=12)
        return r.ok
    except Exception:
        return False


# =========================
# INDICATORS (no pandas_ta)
# =========================
def sma(series: np.ndarray, period: int) -> np.ndarray:
    s = pd.Series(series)
    return s.rolling(period).mean().to_numpy()

def ema(series: np.ndarray, period: int) -> np.ndarray:
    s = pd.Series(series)
    return s.ewm(span=period, adjust=False).mean().to_numpy()

def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    c = pd.Series(close)
    delta = c.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.to_numpy()

def bollinger(close: np.ndarray, period: int = 20, mult: float = 2.0):
    s = pd.Series(close)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    return mid.to_numpy(), upper.to_numpy(), lower.to_numpy()

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.r_[np.nan, close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return tr

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = true_range(high, low, close)
    s = pd.Series(tr)
    return s.ewm(alpha=1/period, adjust=False).mean().to_numpy()

def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    h = pd.Series(high)
    l = pd.Series(low)

    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.Series(true_range(high, low, close))
    atr_w = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan))

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    out = dx.ewm(alpha=1/period, adjust=False).mean()
    return out.to_numpy()


# =========================
# HELPERS
# =========================
def now_istanbul_str():
    ts = datetime.now(timezone.utc).timestamp() + 3 * 3600
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def step_score(x: float, step: int = SCORE_STEP) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    return int(round(x / step) * step)

def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

def symbol_clean(sym: str) -> str:
    return sym.replace(":USDT", "")

def is_usdt_spot_symbol(sym: str) -> bool:
    s = symbol_clean(sym)
    return s.endswith("/USDT") and (":" not in s)

def is_junk_symbol(base: str) -> bool:
    b = base.upper()
    junk = {
        "USDT","USDC","BUSD","TUSD","DAI","FDUSD","USDP","EUR","EURT",
        "3S","3L","5S","5L","BULL","BEAR","UP","DOWN"
    }
    if b in junk:
        return True
    for suf in ["3L","3S","5L","5S","BULL","BEAR","UP","DOWN"]:
        if b.endswith(suf):
            return True
    return False


# =========================
# EXCHANGES
# =========================
@st.cache_resource(show_spinner=False)
def get_exchanges():
    kucoin = ccxt.kucoin({
        "enableRateLimit": CCXT_RATE_LIMIT,
        "timeout": 20000,
        "options": {"defaultType": "spot"},
    })
    okx = ccxt.okx({
        "enableRateLimit": CCXT_RATE_LIMIT,
        "timeout": 20000,
        "options": {"defaultType": "spot"},
    })
    return kucoin, okx

def load_markets_safe(ex, name: str) -> bool:
    try:
        ex.load_markets()
        return True
    except Exception as e:
        st.error(f"{name}: Market yüklenemedi: {e}")
        return False

def fetch_top_usdt_symbols(ex, top_n: int) -> list[str]:
    try:
        tickers = ex.fetch_tickers()
    except Exception:
        tickers = {}

    rows = []
    for sym, t in (tickers or {}).items():
        sym2 = symbol_clean(sym)
        if not is_usdt_spot_symbol(sym2):
            continue

        base = sym2.split("/")[0]
        if is_junk_symbol(base):
            continue

        qv = safe_float(t.get("quoteVolume"), 0.0)
        last = safe_float(t.get("last"), 0.0)
        basev = safe_float(t.get("baseVolume"), 0.0)
        if qv <= 0 and basev > 0 and last > 0:
            qv = basev * last

        bid = safe_float(t.get("bid"), 0.0)
        ask = safe_float(t.get("ask"), 0.0)
        spread_pct = 999.0
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / ((ask + bid) / 2) * 100

        rows.append((sym2, qv, spread_pct))

    df = pd.DataFrame(rows, columns=["symbol","qv24h","spread_pct"])
    if df.empty:
        return []

    df = df[df["qv24h"] >= MIN_QV_24H_USDT]
    df = df[df["spread_pct"] <= MAX_SPREAD_PCT]
    df = df.sort_values("qv24h", ascending=False).head(top_n)
    return df["symbol"].tolist()

def fetch_ohlcv_df(ex, symbol: str, timeframe: str, limit: int = 200):
    try:
        o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not o or len(o) < 120:
            return None
        df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
        return df
    except Exception:
        return None


# =========================
# SCORING (8 GATES) - EŞİT SERT
# =========================
def score_symbol(df: pd.DataFrame, df_htf: pd.DataFrame):
    """
    8 Kapı:
    1) RSI bias:  LONG >= 58  / SHORT <= 42
    2) ADX (TF) >= 22
    3) ATR spike >= 1.08
    4) SMA20 momentum distance: |dist| >= 0.20%
    5) HTF trend: 1H close vs SMA20 aynı yönde
    6) BB/Trend filtresi: LONG mid üstü / SHORT mid altı
    7) HTF ADX >= 20
    8) Mikro kırılım (hareket başladı): LONG close > prev3 high / SHORT close < prev3 low
    """
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    htf_h = df_htf["high"].to_numpy(dtype=float)
    htf_l = df_htf["low"].to_numpy(dtype=float)
    htf_c = df_htf["close"].to_numpy(dtype=float)

    r = rsi(c, 14)
    a = adx(h, l, c, 14)
    mid, up, lowb = bollinger(c, 20, 2.0)
    s20 = sma(c, 20)
    at = atr(h, l, c, 14)

    htf_s20 = sma(htf_c, 20)
    htf_adx = adx(htf_h, htf_l, htf_c, 14)

    if any(np.isnan(x[-1]) for x in [r, a, mid, s20, at, htf_s20, htf_adx]):
        return None

    last = float(c[-1])

    # Direction heuristic (eşit):
    direction = "LONG" if (last >= s20[-1] and r[-1] >= 50) else "SHORT"

    gates = 0

    # 1) RSI bias
    if direction == "LONG" and r[-1] >= 58:
        gates += 1
    if direction == "SHORT" and r[-1] <= 42:
        gates += 1

    # 2) ADX TF
    if a[-1] >= 22:
        gates += 1

    # 3) ATR Spike
    atr_ema = ema(at, 20)
    atr_spike = (at[-1] / atr_ema[-1]) if (atr_ema[-1] and not np.isnan(atr_ema[-1])) else 1.0
    if atr_spike >= 1.08:
        gates += 1

    # 4) SMA20 momentum distance
    sma_dist_pct = (last - s20[-1]) / s20[-1] * 100 if s20[-1] else 0.0
    if direction == "LONG" and sma_dist_pct >= 0.20:
        gates += 1
    if direction == "SHORT" and sma_dist_pct <= -0.20:
        gates += 1

    # 5) HTF Trend
    if direction == "LONG" and htf_c[-1] >= htf_s20[-1]:
        gates += 1
    if direction == "SHORT" and htf_c[-1] <= htf_s20[-1]:
        gates += 1

    # 6) BB / Trend filtresi
    if direction == "LONG" and last >= mid[-1] * 1.001:
        gates += 1
    if direction == "SHORT" and last <= mid[-1] * 0.999:
        gates += 1

    # 7) HTF ADX
    if htf_adx[-1] >= 20:
        gates += 1

    # 8) Mikro kırılım (hareket başladıktan sonra)
    # prev3 high/low
    if len(h) >= 5:
        prev3_high = float(np.max(h[-4:-1]))
        prev3_low = float(np.min(l[-4:-1]))
        if direction == "LONG" and last > prev3_high:
            gates += 1
        if direction == "SHORT" and last < prev3_low:
            gates += 1

    raw = (gates / GATES_TOTAL) * 100.0
    score = step_score(raw, SCORE_STEP)

    # SHORT display mapping: düşük skor = daha iyi short (senin mantık)
    disp_score = score if direction == "LONG" else (100 - score)
    disp_raw = int(round(raw)) if direction == "LONG" else int(round(100 - raw))

    return {
        "direction": direction,
        "gates": int(gates),
        "raw": int(disp_raw),
        "score": int(disp_score),
        "atr_spike": float(atr_spike),
        "rsi": float(r[-1]),
        "adx": float(a[-1]),
        "htf_adx": float(htf_adx[-1]),
        "sma_dist_pct": float(sma_dist_pct),
    }

def strong_flag(direction: str, gates: int, score: int) -> bool:
    if gates < GATES_REQUIRED_STRONG:
        return False
    if direction == "LONG":
        return score >= STRONG_LONG_SCORE_MIN
    return score <= STRONG_SHORT_SCORE_MAX


# =========================
# UI TABLE STYLE (FIXED - HATA YOK)
# =========================
def style_table(df: pd.DataFrame):
    def row_style(row):
        direction = str(row.get("YÖN", ""))
        strong = bool(row.get("STRONG", False))

        # açık/koyu
        if direction == "LONG":
            bg = "#1f6f55" if not strong else "#0f4b37"
        else:
            bg = "#5b1b1b" if not strong else "#3b0f0f"

        css = (
            f"background-color: {bg};"
            "color: #e9f1ef;"
            "border-color: rgba(255,255,255,0.10);"
            f"font-weight: {'700' if strong else '500'};"
        )
        # KRİTİK: kolon sayısı kadar döndür
        return [css] * len(row)

    styler = df.style.apply(row_style, axis=1)

    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [
                ("background-color", "#121826"),
                ("color", "#dbe7ff"),
                ("border-color", "rgba(255,255,255,0.12)"),
                ("font-weight", "700"),
            ]},
            {"selector": "td", "props": [("border-color", "rgba(255,255,255,0.10)")]},
        ]
    )

    fmt = {}
    if "FİYAT" in df.columns:
        fmt["FİYAT"] = "{:,.6f}"
    if "QV_24H" in df.columns:
        fmt["QV_24H"] = "{:,.0f}"
    if fmt:
        styler = styler.format(fmt, na_rep="-")

    return styler


# =========================
# TELEGRAM MESSAGE BUILDERS (TEK MESAJ, MAX 20)
# =========================
def _tg_line(coin: str, score: int, direction: str, src: str, is_long_block: bool) -> str:
    bullet = "🟩" if is_long_block else "🟥"
    dir_emoji = "🟢" if direction == "LONG" else "🔴"
    return f"{bullet} {coin}: <b>{score}</b> Puan | {dir_emoji} <b>{direction}</b>  <i>({src})</i>"

def _pick_top_for_telegram(df_all: pd.DataFrame, only_strong_both: bool) -> pd.DataFrame:
    df = df_all.copy()
    if only_strong_both:
        df = df[(df["STRONG"] == True) & (df["SOURCE"] == "BOTH")]

    if df.empty:
        return df

    df["_prio_both"] = (df["SOURCE"] == "BOTH").astype(int)
    df["_prio_strong"] = (df["STRONG"] == True).astype(int)

    # ortak sıralama: LONG yüksek iyi, SHORT düşük iyi
    df["_score_rank"] = np.where(df["YÖN"] == "SHORT", 100 - df["SKOR"], df["SKOR"])

    df = df.sort_values(
        by=["_prio_both", "_prio_strong", "KAPI", "_score_rank", "QV_24H"],
        ascending=[False, False, False, False, False]
    ).drop(columns=["_prio_both", "_prio_strong", "_score_rank"])

    return df.head(TG_MAX_LIST)

def tg_send_report_list(df_list: pd.DataFrame):
    lines = []
    lines.append("📋 <b>Gözcü Raporu (L/S)</b>")
    lines.append(f"⏱ İstanbul: <code>{now_istanbul_str()}</code>")
    lines.append("")

    if df_list.empty:
        lines.append("⚠️ Aday yok. Bir sonraki yenilemede tekrar dene.")
        send_telegram_html("\n".join(lines))
        return

    df_long = df_list[df_list["YÖN"] == "LONG"]
    df_short = df_list[df_list["YÖN"] == "SHORT"]

    for _, r in df_long.iterrows():
        lines.append(_tg_line(r["COIN"], int(r["SKOR"]), r["YÖN"], r["SOURCE"], True))
    for _, r in df_short.iterrows():
        lines.append(_tg_line(r["COIN"], int(r["SKOR"]), r["YÖN"], r["SOURCE"], False))

    send_telegram_html("\n".join(lines))


# =========================
# MAIN UI
# =========================
st.title("🎯 Sniper — Auto (LONG + SHORT)")
st.caption(
    f"TF={TF} • HTF={HTF} • STRONG: 8/8 Kapı (çok seçici) • "
    f"Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s"
)

st_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")

if "sent_keys" not in st.session_state:
    st.session_state.sent_keys = set()
if "last_report_ts" not in st.session_state:
    st.session_state.last_report_ts = 0.0

kucoin, okx = get_exchanges()

colA, colB = st.columns(2)
with colA:
    ku_ok = load_markets_safe(kucoin, "KuCoin")
    st.success("KuCoin: ✅ Bağlandı" if ku_ok else "KuCoin: ❌ Hata")
with colB:
    ok_ok = load_markets_safe(okx, "OKX")
    st.success("OKX: ✅ Bağlandı" if ok_ok else "OKX: ❌ Hata")

if tg_enabled():
    st.info("Telegram: ✅ Secrets bulundu (TG_TOKEN + TG_CHAT_ID).")
else:
    st.warning("Telegram: ❌ Secrets eksik. (TG_TOKEN / TG_CHAT_ID)")

if not (ku_ok and ok_ok):
    st.stop()

with st.spinner("Marketler taranıyor (hacim/spread filtreleri uygulanıyor)..."):
    ku_syms = fetch_top_usdt_symbols(kucoin, TOP_N_PER_EXCHANGE)
    ok_syms = fetch_top_usdt_symbols(okx, TOP_N_PER_EXCHANGE)

ku_set = set(ku_syms)
ok_set = set(ok_syms)
scan_list = sorted(list(ku_set.union(ok_set)))

progress = st.progress(0, text=f"Taranıyor: 0/{len(scan_list)}")

rows = []
strong_rows = []

for i, sym in enumerate(scan_list, start=1):
    progress.progress(i / max(1, len(scan_list)), text=f"Taranıyor: {i}/{len(scan_list)} • {sym}")

    base = sym.split("/")[0]
    res_ku = None
    res_ok = None

    if sym in ku_set:
        df = fetch_ohlcv_df(kucoin, sym, TF, 200)
        dfh = fetch_ohlcv_df(kucoin, sym, HTF, 200)
        if df is not None and dfh is not None:
            res_ku = score_symbol(df, dfh)

    if sym in ok_set:
        df = fetch_ohlcv_df(okx, sym, TF, 200)
        dfh = fetch_ohlcv_df(okx, sym, HTF, 200)
        if df is not None and dfh is not None:
            res_ok = score_symbol(df, dfh)

    source = None
    chosen = None

    if res_ku and res_ok:
        if res_ku["direction"] == res_ok["direction"]:
            direction = res_ku["direction"]
            gates = min(res_ku["gates"], res_ok["gates"])
            score = int(round((res_ku["score"] + res_ok["score"]) / 2))
            raw = int(round((res_ku["raw"] + res_ok["raw"]) / 2))
            chosen = {"direction": direction, "gates": gates, "score": score, "raw": raw}
            source = "BOTH"
        else:
            # ayrışma: kapısı yüksek olanı seç
            pick = res_ku if res_ku["gates"] > res_ok["gates"] else res_ok
            chosen = pick
            source = "BOTH"
    elif res_ku:
        chosen = res_ku
        source = "KUCOIN"
    elif res_ok:
        chosen = res_ok
        source = "OKX"
    else:
        continue

    last_price = 0.0
    qv = 0.0
    try:
        if source in ("KUCOIN", "BOTH"):
            t = kucoin.fetch_ticker(sym)
            last_price = safe_float(t.get("last"), last_price)
            qv = max(qv, safe_float(t.get("quoteVolume"), 0.0))
        if source in ("OKX", "BOTH"):
            t = okx.fetch_ticker(sym)
            last_price = max(last_price, safe_float(t.get("last"), last_price))
            qv = max(qv, safe_float(t.get("quoteVolume"), 0.0))
    except Exception:
        pass

    direction = chosen["direction"]
    gates = int(chosen["gates"])
    score = int(chosen["score"])
    raw = int(chosen["raw"])
    strong = strong_flag(direction, gates, score)

    row = {
        "YÖN": direction,
        "COIN": base,
        "SKOR": score,
        "FİYAT": last_price,
        "RAW": raw,
        "QV_24H": int(qv) if qv else 0,
        "KAPI": gates,
        "STRONG": strong,
        "SOURCE": source,
    }
    rows.append(row)

    if strong and source == "BOTH":
        strong_rows.append(row)

progress.empty()

df_all = pd.DataFrame(rows)
if df_all.empty:
    st.warning("Hiç aday çıkmadı. Filtreler çok sert olabilir (hacim/spread).")
    st.stop()

# UI sıralama: BOTH + STRONG + KAPI + skor
df_all["_prio_both"] = (df_all["SOURCE"] == "BOTH").astype(int)
df_all["_prio_strong"] = (df_all["STRONG"] == True).astype(int)
df_all["_score_rank"] = np.where(df_all["YÖN"] == "SHORT", 100 - df_all["SKOR"], df_all["SKOR"])

df_all = df_all.sort_values(
    by=["_prio_both", "_prio_strong", "KAPI", "_score_rank", "QV_24H"],
    ascending=[False, False, False, False, False]
).drop(columns=["_prio_both", "_prio_strong", "_score_rank"])

strong_long = int(((df_all["STRONG"]) & (df_all["YÖN"] == "LONG") & (df_all["SOURCE"] == "BOTH")).sum())
strong_short = int(((df_all["STRONG"]) & (df_all["YÖN"] == "SHORT") & (df_all["SOURCE"] == "BOTH")).sum())
cnt_long = int((df_all["YÖN"] == "LONG").sum())
cnt_short = int((df_all["YÖN"] == "SHORT").sum())

st.markdown(f"**İstanbul Time:** `{now_istanbul_str()}`")
st.success("Tarama bitti ✅")
st.info(f"✅ STRONG LONG: {strong_long} | 💀 STRONG SHORT: {strong_short} | LONG: {cnt_long} | SHORT: {cnt_short}")

# UI tablo: 10 LONG + 10 SHORT
df_long = df_all[df_all["YÖN"] == "LONG"].head(TABLE_N_LONG)
df_short = df_all[df_all["YÖN"] == "SHORT"].head(TABLE_N_SHORT)
df_show = pd.concat([df_long, df_short], ignore_index=True)

if strong_long + strong_short == 0:
    st.warning("⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu.")
else:
    st.success("✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu.")

st.subheader("🎯 SNIPER TABLO")
st.dataframe(
    style_table(df_show),
    use_container_width=True,
    hide_index=True
)


# =========================
# TELEGRAM LOGIC
# =========================
# STRONG varsa: TEK MESAJ + STRONG (BOTH) en iyi max 20
# STRONG yoksa: 20 dakikada bir TEK MESAJ + en iyi max 20
if tg_enabled():
    # STRONG best list
    df_strongs_best = _pick_top_for_telegram(df_all, only_strong_both=True)

    # Dedupe: aynı coin/yön/skor/kapı tekrar yollamasın
    new_keys = []
    if not df_strongs_best.empty:
        for _, r in df_strongs_best.iterrows():
            k = f"{r['COIN']}|{r['YÖN']}|{r['SOURCE']}|{r['SKOR']}|{r['KAPI']}"
            if k not in st.session_state.sent_keys:
                st.session_state.sent_keys.add(k)
                new_keys.append(k)

        # bu refresh'te yeni strong yakalandıysa -> tek mesaj
        if new_keys:
            tg_send_report_list(df_strongs_best)

    # Yeni strong yoksa -> periyodik rapor
    if not new_keys:
        now_ts = time.time()
        if (now_ts - st.session_state.last_report_ts) >= (REPORT_EVERY_MIN * 60):
            df_best20 = _pick_top_for_telegram(df_all, only_strong_both=False)
            tg_send_report_list(df_best20)
            st.session_state.last_report_ts = now_ts

with st.expander("🧪 Telegram Test (İstersen)"):
    if tg_enabled():
        if st.button("Test Mesajı Gönder"):
            ok = send_telegram_html(
                "✅ <b>Sniper Gözcü</b> Telegram testi çalışıyor.\n"
                f"⏱ İstanbul: <code>{now_istanbul_str()}</code>"
            )
            st.success("Gönderildi ✅" if ok else "Gönderilemedi ❌")
    else:
        st.warning("Secrets yok: TG_TOKEN / TG_CHAT_ID eklemeden test olmaz.")
