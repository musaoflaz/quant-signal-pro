# requirements.txt (install these):
# streamlit
# pandas
# numpy
# ccxt
# requests
# streamlit-autorefresh

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import ccxt


# =========================
# CONFIG (NO MANUAL SETTINGS)
# =========================
st.set_page_config(page_title="Sniper — Auto (LONG + SHORT)", layout="wide")

TF = "15m"
HTF = "1h"

TOP_N_PER_EXCHANGE = 180          # KuCoin + OKX (likid filtre sonrası)
TABLE_N_LONG = 10
TABLE_N_SHORT = 10

AUTO_REFRESH_SEC = 240            # Streamlit auto refresh (test)
REPORT_EVERY_MIN = 20             # STRONG yoksa rapor kaç dakikada bir Telegram

# 8/8 FULL SNIPER MOD (SERT)
GATES_REQUIRED_STRONG = 8
SCORE_STEP = 2                    # skor 2'şer
STRONG_LONG_SCORE_MIN = 100        # 8/8 LONG -> 100
STRONG_SHORT_SCORE_MAX = 0         # 8/8 SHORT -> 0

# Likidite filtresi (çöp azaltma)
MIN_QV_24H_USDT = 300_000
MAX_SPREAD_PCT = 0.35

# Telegram list max
TG_MAX_LIST = 20

# ccxt
CCXT_RATE_LIMIT = True
CANDLE_LIMIT = 220


# =========================
# DARK THEME (UI FIX)
# =========================
st.markdown(
    """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
h1, h2, h3, h4, h5, h6, p, span, div, label { color: #e6edf3 !important; }
[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
[data-testid="stSidebar"] { display: none; }
div[data-testid="stStatusWidget"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


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
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
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
    return s.ewm(alpha=1 / period, adjust=False).mean().to_numpy()


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    h = pd.Series(high)
    l = pd.Series(low)

    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.Series(true_range(high, low, close))
    atr_w = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w.replace(0, np.nan))

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    out = dx.ewm(alpha=1 / period, adjust=False).mean()
    return out.to_numpy()


# =========================
# HELPERS
# =========================
def now_istanbul_str():
    ts = datetime.now(timezone.utc).timestamp() + 3 * 3600  # IST fixed UTC+3
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


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
        "USDT",
        "USDC",
        "BUSD",
        "TUSD",
        "DAI",
        "FDUSD",
        "USDP",
        "EUR",
        "EURT",
        "3S",
        "3L",
        "5S",
        "5L",
        "BULL",
        "BEAR",
        "UP",
        "DOWN",
    }
    if b in junk:
        return True
    for suf in ["3L", "3S", "5L", "5S", "BULL", "BEAR", "UP", "DOWN"]:
        if b.endswith(suf):
            return True
    return False


def pct_change(a: float, b: float) -> float:
    if a == 0 or a is None or b is None:
        return 0.0
    return (b - a) / a * 100.0


# =========================
# EXCHANGES
# =========================
@st.cache_resource(show_spinner=False)
def get_exchanges():
    kucoin = ccxt.kucoin(
        {
            "enableRateLimit": CCXT_RATE_LIMIT,
            "timeout": 20000,
            "options": {"defaultType": "spot"},
        }
    )
    okx = ccxt.okx(
        {
            "enableRateLimit": CCXT_RATE_LIMIT,
            "timeout": 20000,
            "options": {"defaultType": "spot"},
        }
    )
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

    df = pd.DataFrame(rows, columns=["symbol", "qv24h", "spread_pct"])
    if df.empty:
        return []

    df = df[df["qv24h"] >= MIN_QV_24H_USDT]
    df = df[df["spread_pct"] <= MAX_SPREAD_PCT]
    df = df.sort_values("qv24h", ascending=False).head(top_n)
    return df["symbol"].tolist()


def fetch_ohlcv_df(ex, symbol: str, timeframe: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame | None:
    try:
        o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not o or len(o) < 100:
            return None
        return pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
    except Exception:
        return None


# =========================
# MARKET REGIME (BTC FILTER / INFO)
# =========================
def btc_regime(okx, kucoin) -> dict:
    """
    Professional advice: calculate BTC 1H regime.
    We DO NOT block sides (B mode). We only label ALIGN and count "Aligned Strong".
    """
    sym = "BTC/USDT"
    df = fetch_ohlcv_df(okx, sym, HTF, CANDLE_LIMIT)
    src = "OKX"
    if df is None:
        df = fetch_ohlcv_df(kucoin, sym, HTF, CANDLE_LIMIT)
        src = "KUCOIN"

    if df is None:
        return {"src": "N/A", "state": "UNKNOWN", "close": np.nan, "sma20": np.nan, "rsi": np.nan}

    c = df["close"].to_numpy(dtype=float)
    s20 = sma(c, 20)
    r = rsi(c, 14)

    last = float(c[-1])
    last_sma = float(s20[-1]) if not np.isnan(s20[-1]) else np.nan
    last_rsi = float(r[-1]) if not np.isnan(r[-1]) else np.nan

    state = "NEUTRAL"
    if not np.isnan(last_sma) and not np.isnan(last_rsi):
        if last > last_sma and last_rsi >= 50:
            state = "BULL"
        elif last < last_sma and last_rsi <= 50:
            state = "BEAR"

    return {"src": src, "state": state, "close": last, "sma20": last_sma, "rsi": last_rsi}


# =========================
# SCORING (8 GATES - FULL SERT)
# =========================
def score_symbol_8gates(df: pd.DataFrame, df_htf: pd.DataFrame) -> dict | None:
    """
    Returns dict:
      direction: LONG/SHORT
      gates: 0..8
      raw: display raw (SHORT inverted)
      score: display score stepped (SHORT inverted)
      details: optional
    """
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    hh = df_htf["high"].to_numpy(dtype=float)
    hl = df_htf["low"].to_numpy(dtype=float)
    hc = df_htf["close"].to_numpy(dtype=float)

    r = rsi(c, 14)
    a = adx(h, l, c, 14)
    mid, up, lowb = bollinger(c, 20, 2.0)
    s20 = sma(c, 20)
    at = atr(h, l, c, 14)

    htf_sma20 = sma(hc, 20)
    htf_adx = adx(hh, hl, hc, 14)

    if any(
        np.isnan(x[-1])
        for x in [r, a, mid, up, lowb, s20, at, htf_sma20, htf_adx]
    ):
        return None

    last = float(c[-1])
    last_rsi = float(r[-1])
    last_adx = float(a[-1])
    last_mid = float(mid[-1])
    last_up = float(up[-1])
    last_low = float(lowb[-1])
    last_sma = float(s20[-1])
    last_atr = float(at[-1])

    last_htf_close = float(hc[-1])
    last_htf_sma = float(htf_sma20[-1])
    last_htf_adx = float(htf_adx[-1])

    # Direction heuristic (strict)
    direction = "LONG" if (last >= last_sma and last_rsi >= 50) else "SHORT"

    gates = 0

    # Gate 1: RSI bias (strict)
    if direction == "LONG" and last_rsi >= 60:
        gates += 1
    if direction == "SHORT" and last_rsi <= 40:
        gates += 1

    # Gate 2: ADX strength (strict)
    if last_adx >= 25:
        gates += 1

    # Gate 3: Bollinger strength position (strict near band, away from mid)
    # LONG: above mid and closer to upper band
    # SHORT: below mid and closer to lower band
    if direction == "LONG":
        if last >= last_mid * 1.003 and (last_up > last_mid) and ((last_up - last) / (last_up - last_mid + 1e-9) <= 0.35):
            gates += 1
    else:
        if last <= last_mid * 0.997 and (last_mid > last_low) and ((last - last_low) / (last_mid - last_low + 1e-9) <= 0.35):
            gates += 1

    # Gate 4: SMA20 momentum distance (strict)
    sma_dist_pct = (last - last_sma) / (last_sma + 1e-9) * 100.0
    if direction == "LONG" and sma_dist_pct >= 0.35:
        gates += 1
    if direction == "SHORT" and sma_dist_pct <= -0.35:
        gates += 1

    # Gate 5: ATR spike (strict)
    atr_ema = ema(at, 20)
    last_atr_ema = float(atr_ema[-1]) if not np.isnan(atr_ema[-1]) else np.nan
    atr_spike = (last_atr / (last_atr_ema + 1e-9)) if not np.isnan(last_atr_ema) else 1.0
    if atr_spike >= 1.15:
        gates += 1

    # Gate 6: HTF trend confirmation (strict)
    if direction == "LONG" and last_htf_close >= last_htf_sma * 1.001:
        gates += 1
    if direction == "SHORT" and last_htf_close <= last_htf_sma * 0.999:
        gates += 1

    # Gate 7: HTF ADX (strict)
    if last_htf_adx >= 23:
        gates += 1

    # Gate 8: ENTRY QUALITY (SERT - geç giriş yok)
    # - Last 3 candles pump/dump must be limited
    # - Not too stretched from SMA20
    # - Avoid extreme RSI
    if len(c) >= 5:
        ch3 = pct_change(float(c[-4]), float(c[-1]))
    else:
        ch3 = 0.0

    stretch_ok = abs(sma_dist_pct) <= 1.50

    if direction == "LONG":
        entry_ok = (last_rsi <= 70) and stretch_ok and (ch3 <= 3.0)
        if entry_ok:
            gates += 1
    else:
        # For short: avoid extreme oversold and avoid "late dump"
        entry_ok = (last_rsi >= 30) and stretch_ok and (ch3 >= -3.0)
        if entry_ok:
            gates += 1

    raw = (gates / 8.0) * 100.0
    score = step_score(raw, SCORE_STEP)

    # SHORT display mapping: low score = better short (your style)
    disp_score = score if direction == "LONG" else (100 - score)
    disp_raw = int(round(raw)) if direction == "LONG" else int(round(100 - raw))

    return {
        "direction": direction,
        "gates": int(gates),
        "raw": int(disp_raw),
        "score": int(disp_score),
        "sma_dist_pct": float(sma_dist_pct),
        "atr_spike": float(atr_spike),
        "rsi": float(last_rsi),
        "adx": float(last_adx),
        "htf_adx": float(last_htf_adx),
    }


def strong_flag(direction: str, gates: int, score: int) -> bool:
    if gates < GATES_REQUIRED_STRONG:
        return False
    if direction == "LONG":
        return score >= STRONG_LONG_SCORE_MIN
    return score <= STRONG_SHORT_SCORE_MAX


# =========================
# UI TABLE STYLE (OLD LOOK: LONG green / SHORT red / STRONG darker)
# =========================
def style_table(df: pd.DataFrame):
    def row_style(row):
        direction = str(row.get("YÖN", ""))
        strong = bool(row.get("STRONG", False))
        both = (str(row.get("SOURCE", "")) == "BOTH")

        if direction == "LONG":
            bg = "#1f6f55" if not strong else "#0f4b37"   # green / darker green
        else:
            bg = "#5b1b1b" if not strong else "#3b0f0f"   # red / darker red

        # BOTH + STRONG slightly more emphasis via border/glow-ish (still simple)
        border = "rgba(255,255,255,0.10)" if (both and strong) else "rgba(255,255,255,0.06)"
        return [
            f"background-color: {bg}",
            "color: #e9f1ef",
            f"border-color: {border}",
        ]

    styler = df.style.apply(lambda r: row_style(r), axis=1)
    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("background-color", "#121826"), ("color", "#dbe7ff"), ("border-color", "rgba(255,255,255,0.12)")]},
            {"selector": "td", "props": [("border-color", "rgba(255,255,255,0.06)")]},
        ]
    )
    styler = styler.format({"FİYAT": "{:,.6f}", "QV_24H": "{:,.0f}"}, na_rep="-")
    return styler


# =========================
# TELEGRAM MESSAGE (SINGLE LIST, LIKE YOUR SCREENSHOTS)
# =========================
def _tg_line(coin: str, score: int, direction: str, src: str, align: bool) -> str:
    bullet = "🟩" if direction == "LONG" else "🟥"
    dir_emoji = "🟢" if direction == "LONG" else "🔴"
    align_mark = "✅" if align else "⚪"
    return f"{bullet} {align_mark} {coin}: <b>{score}</b> Puan | {dir_emoji} <b>{direction}</b>  <i>({src})</i>"


def pick_best_for_telegram(df_all: pd.DataFrame, only_strong_both: bool) -> pd.DataFrame:
    df = df_all.copy()
    if only_strong_both:
        df = df[(df["STRONG"] == True) & (df["SOURCE"] == "BOTH")]

    if df.empty:
        return df

    df["_prio_both"] = (df["SOURCE"] == "BOTH").astype(int)
    df["_prio_strong"] = (df["STRONG"] == True).astype(int)
    df["_prio_align"] = (df["ALIGN"] == True).astype(int)
    df["_score_rank"] = np.where(df["YÖN"] == "SHORT", 100 - df["SKOR"], df["SKOR"])

    df = df.sort_values(
        by=["_prio_both", "_prio_strong", "_prio_align", "KAPI", "_score_rank", "QV_24H"],
        ascending=[False, False, False, False, False, False],
    ).drop(columns=["_prio_both", "_prio_strong", "_prio_align", "_score_rank"])

    return df.head(TG_MAX_LIST)


def tg_send_report(df_list: pd.DataFrame, title: str):
    lines = []
    lines.append(f"📋 <b>{title}</b>")
    lines.append(f"⏱ İstanbul: <code>{now_istanbul_str()}</code>")
    lines.append("")

    if df_list is None or df_list.empty:
        lines.append("⚠️ Aday yok. Bir sonraki yenilemede tekrar dene.")
        send_telegram_html("\n".join(lines))
        return

    df_long = df_list[df_list["YÖN"] == "LONG"]
    df_short = df_list[df_list["YÖN"] == "SHORT"]

    for _, r in df_long.iterrows():
        lines.append(_tg_line(r["COIN"], int(r["SKOR"]), r["YÖN"], r["SOURCE"], bool(r.get("ALIGN", False))))
    for _, r in df_short.iterrows():
        lines.append(_tg_line(r["COIN"], int(r["SKOR"]), r["YÖN"], r["SOURCE"], bool(r.get("ALIGN", False))))

    send_telegram_html("\n".join(lines))


# =========================
# MAIN UI
# =========================
st.title("🎯 Sniper — Auto (LONG + SHORT)")
st.caption(
    f"TF={TF} • HTF={HTF} • FULL SERT: STRONG = {GATES_REQUIRED_STRONG}/{GATES_REQUIRED_STRONG} • "
    f"Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s"
)

st_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")

if "sent_keys" not in st.session_state:
    st.session_state.sent_keys = set()
if "last_report_ts" not in st.session_state:
    st.session_state.last_report_ts = 0.0

kucoin, okx = get_exchanges()

# Connection cards
colA, colB, colC = st.columns([1, 1, 1])
with colA:
    ku_ok = load_markets_safe(kucoin, "KuCoin")
    st.success("KuCoin: ✅ Bağlandı" if ku_ok else "KuCoin: ❌ Hata")
with colB:
    ok_ok = load_markets_safe(okx, "OKX")
    st.success("OKX: ✅ Bağlandı" if ok_ok else "OKX: ❌ Hata")
with colC:
    if tg_enabled():
        st.info("Telegram: ✅ Secrets OK")
    else:
        st.warning("Telegram: ❌ Secrets eksik")

if not (ku_ok and ok_ok):
    st.stop()

# BTC regime (info only, professional overlay)
reg = btc_regime(okx, kucoin)
reg_state = reg["state"]
reg_src = reg["src"]
st.markdown(
    f"**İstanbul Time:** `{now_istanbul_str()}`  •  **BTC Regime (1H):** `{reg_state}`  •  Source: `{reg_src}`"
)

# Load symbols
with st.spinner("Marketler taranıyor (hacim/spread filtreleri uygulanıyor)..."):
    ku_syms = fetch_top_usdt_symbols(kucoin, TOP_N_PER_EXCHANGE)
    ok_syms = fetch_top_usdt_symbols(okx, TOP_N_PER_EXCHANGE)

ku_set = set(ku_syms)
ok_set = set(ok_syms)
scan_list = sorted(list(ku_set.union(ok_set)))

# Scan
progress = st.progress(0, text=f"Taranıyor: 0/{len(scan_list)}")

rows = []
strong_rows = []

for i, sym in enumerate(scan_list, start=1):
    progress.progress(i / max(1, len(scan_list)), text=f"Taranıyor: {i}/{len(scan_list)} • {sym}")

    base = sym.split("/")[0]

    res_ku = None
    res_ok = None

    if sym in ku_set:
        df = fetch_ohlcv_df(kucoin, sym, TF, CANDLE_LIMIT)
        dfh = fetch_ohlcv_df(kucoin, sym, HTF, CANDLE_LIMIT)
        if df is not None and dfh is not None:
            res_ku = score_symbol_8gates(df, dfh)

    if sym in ok_set:
        df = fetch_ohlcv_df(okx, sym, TF, CANDLE_LIMIT)
        dfh = fetch_ohlcv_df(okx, sym, HTF, CANDLE_LIMIT)
        if df is not None and dfh is not None:
            res_ok = score_symbol_8gates(df, dfh)

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
            # Divergence: pick higher gates then better "strength"
            a = res_ku
            b = res_ok
            if a["gates"] != b["gates"]:
                pick = a if a["gates"] > b["gates"] else b
            else:
                # compare strength rank (LONG: higher score better, SHORT: lower better)
                a_rank = (a["score"] if a["direction"] == "LONG" else (100 - a["score"]))
                b_rank = (b["score"] if b["direction"] == "LONG" else (100 - b["score"]))
                pick = a if a_rank >= b_rank else b
            chosen = {"direction": pick["direction"], "gates": pick["gates"], "score": pick["score"], "raw": pick["raw"]}
            source = "BOTH"
    elif res_ku:
        chosen = {"direction": res_ku["direction"], "gates": res_ku["gates"], "score": res_ku["score"], "raw": res_ku["raw"]}
        source = "KUCOIN"
    elif res_ok:
        chosen = {"direction": res_ok["direction"], "gates": res_ok["gates"], "score": res_ok["score"], "raw": res_ok["raw"]}
        source = "OKX"
    else:
        continue

    # Price + qv best-effort
    last_price = 0.0
    qv = 0.0
    try:
        if source in ("KUCOIN", "BOTH"):
            t = kucoin.fetch_ticker(sym)
            last_price = safe_float(t.get("last"), last_price)
            qv = max(qv, safe_float(t.get("quoteVolume"), 0.0))
        if source in ("OKX", "BOTH"):
            t = okx.fetch_ticker(sym)
            last_price = max(last_price, safe_float(t.get("last"), 0.0))
            qv = max(qv, safe_float(t.get("quoteVolume"), 0.0))
    except Exception:
        pass

    direction = chosen["direction"]
    gates = int(chosen["gates"])
    score = int(chosen["score"])
    raw = int(chosen["raw"])
    strong = strong_flag(direction, gates, score)

    # BTC regime alignment (does NOT block, B mode)
    align = False
    if reg_state == "BULL" and direction == "LONG":
        align = True
    if reg_state == "BEAR" and direction == "SHORT":
        align = True
    if reg_state == "NEUTRAL":
        align = True  # neutral = allow both

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
        "ALIGN": align,
    }
    rows.append(row)

    # strong rows for telegram trigger (only BOTH strong)
    if strong and source == "BOTH":
        strong_rows.append(row)

progress.empty()

df_all = pd.DataFrame(rows)
if df_all.empty:
    st.warning("Hiç aday çıkmadı. Filtreler çok sert olabilir (hacim/spread).")
    st.stop()

# Sort priority: BOTH -> STRONG -> ALIGN -> KAPI -> SCORE_RANK -> QV
df_all["_prio_both"] = (df_all["SOURCE"] == "BOTH").astype(int)
df_all["_prio_strong"] = (df_all["STRONG"] == True).astype(int)
df_all["_prio_align"] = (df_all["ALIGN"] == True).astype(int)
df_all["_score_rank"] = np.where(df_all["YÖN"] == "SHORT", 100 - df_all["SKOR"], df_all["SKOR"])

df_all = df_all.sort_values(
    by=["_prio_both", "_prio_strong", "_prio_align", "KAPI", "_score_rank", "QV_24H"],
    ascending=[False, False, False, False, False, False],
).drop(columns=["_prio_both", "_prio_strong", "_prio_align", "_score_rank"])

# Counters
strong_long = int(((df_all["STRONG"]) & (df_all["YÖN"] == "LONG") & (df_all["SOURCE"] == "BOTH")).sum())
strong_short = int(((df_all["STRONG"]) & (df_all["YÖN"] == "SHORT") & (df_all["SOURCE"] == "BOTH")).sum())
cnt_long = int((df_all["YÖN"] == "LONG").sum())
cnt_short = int((df_all["YÖN"] == "SHORT").sum())

aligned_strong_long = int(((df_all["STRONG"]) & (df_all["YÖN"] == "LONG") & (df_all["SOURCE"] == "BOTH") & (df_all["ALIGN"] == True)).sum())
aligned_strong_short = int(((df_all["STRONG"]) & (df_all["YÖN"] == "SHORT") & (df_all["SOURCE"] == "BOTH") & (df_all["ALIGN"] == True)).sum())

# UI header stats
st.success("Tarama bitti ✅")
st.info(
    f"✅ STRONG LONG(BOTH): {strong_long} | 💀 STRONG SHORT(BOTH): {strong_short} | "
    f"LONG: {cnt_long} | SHORT: {cnt_short}  •  "
    f"🎯 Aligned STRONG (BTC Regime): LONG {aligned_strong_long} / SHORT {aligned_strong_short}"
)

# Always show 20 rows (10 long + 10 short) filled by top ranking
df_long = df_all[df_all["YÖN"] == "LONG"].head(TABLE_N_LONG)
df_short = df_all[df_all["YÖN"] == "SHORT"].head(TABLE_N_SHORT)
df_show = pd.concat([df_long, df_short], ignore_index=True)

if (strong_long + strong_short) == 0:
    st.warning("⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu.")
else:
    st.success("✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu.")

st.subheader("🎯 SNIPER TABLO")
st.dataframe(style_table(df_show), use_container_width=True, hide_index=True)


# =========================
# TELEGRAM LOGIC (SINGLE LIST)
# - If NEW STRONG (BOTH) appears -> send single list (best strongs)
# - Else every 20 min -> send report (best 20)
# =========================
if tg_enabled():
    df_strongs = pd.DataFrame(strong_rows) if strong_rows else pd.DataFrame(columns=df_all.columns)

    new_keys = []
    if not df_strongs.empty:
        best_strongs = pick_best_for_telegram(df_all, only_strong_both=True)

        if not best_strongs.empty:
            for _, r in best_strongs.iterrows():
                k = f"{r['COIN']}|{r['YÖN']}|{r['SOURCE']}|{int(r['SKOR'])}|{int(r['KAPI'])}"
                if k not in st.session_state.sent_keys:
                    st.session_state.sent_keys.add(k)
                    new_keys.append(k)

            if new_keys:
                tg_send_report(best_strongs, "STRONG SİNYAL (8/8) — BOTH")

    # No new strong -> periodic report
    if not new_keys:
        now_ts = time.time()
        if (now_ts - st.session_state.last_report_ts) >= (REPORT_EVERY_MIN * 60):
            best20 = pick_best_for_telegram(df_all, only_strong_both=False)
            tg_send_report(best20, "Gözcü Raporu (L/S)")
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
