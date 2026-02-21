# requirements.txt (install these):
# streamlit
# pandas
# numpy
# requests
# ccxt
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
# CONFIG (AUTO / NO SIDEBAR)
# =========================
st.set_page_config(page_title="Sniper — Auto (LONG + SHORT)", layout="wide")

# Timeframes
TF = "15m"
HTF = "1h"

# Universe (liquid + low spread filtered first, then scan)
TOP_N_PER_EXCHANGE = 180          # each exchange: scan top N after liquidity/spread filter

# UI table (always 20 rows total)
TABLE_N_LONG = 10
TABLE_N_SHORT = 10

# Auto refresh (Streamlit Cloud test)
AUTO_REFRESH_SEC = 240           # 4 min refresh

# Telegram report cadence if no NEW strong
REPORT_EVERY_MIN = 20

# Liquidity / quality filters
MIN_QV_24H_USDT = 300_000
MAX_SPREAD_PCT = 0.35            # percent

# Rate limit
CCXT_RATE_LIMIT = True

# Momentum Sniper strictness (EQUAL for LONG/SHORT)
# 8/8 gates required for STRONG
GATES_TOTAL = 8
GATES_REQUIRED_STRONG = 8

# Score quantization (2-by-2)
SCORE_STEP = 2

# STRONG thresholds (very strict)
STRONG_LONG_SCORE_MIN = 98       # 98/100
STRONG_SHORT_SCORE_MAX = 2       # 0/2

# Gates thresholds (momentum mode)
RSI_TF_LONG_MIN = 58
RSI_TF_SHORT_MAX = 42
ADX_MIN = 25
ATR_SPIKE_MIN = 1.15
HTF_RSI_LONG_MIN = 55
HTF_RSI_SHORT_MAX = 45
SMA_DIST_PCT_MIN = 0.25          # 0.25%
BB_MID_BUFFER = 0.001            # 0.1% away from mid

# Telegram max list lines (in one message)
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
def now_istanbul_str() -> str:
    # Istanbul is UTC+3 constant (no DST)
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
    # OKX may return "BTC/USDT:USDT" etc
    return sym.replace(":USDT", "")


def is_usdt_spot_symbol(sym: str) -> bool:
    s = symbol_clean(sym)
    return s.endswith("/USDT") and (":" not in s)


def is_junk_symbol(base: str) -> bool:
    b = base.upper()
    junk = {
        "USDT", "USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "EUR", "EURT",
        "3S", "3L", "5S", "5L", "BULL", "BEAR", "UP", "DOWN"
    }
    if b in junk:
        return True
    for suf in ["3L", "3S", "5L", "5S", "BULL", "BEAR", "UP", "DOWN"]:
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


def fetch_tickers_safe(ex) -> dict:
    try:
        return ex.fetch_tickers() or {}
    except Exception:
        return {}


def fetch_top_usdt_symbols_and_tickers(ex, top_n: int):
    """
    Returns: (symbols_list, tickers_dict_cleaned)
    """
    tickers = fetch_tickers_safe(ex)
    rows = []

    # Normalize keys as cleaned spot symbols for internal lookup
    tickers_clean = {}
    for sym, t in (tickers or {}).items():
        sym2 = symbol_clean(sym)
        tickers_clean[sym2] = t

    for sym2, t in tickers_clean.items():
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
        return [], tickers_clean

    df = df[df["qv24h"] >= MIN_QV_24H_USDT]
    df = df[df["spread_pct"] <= MAX_SPREAD_PCT]
    df = df.sort_values("qv24h", ascending=False).head(top_n)

    return df["symbol"].tolist(), tickers_clean


def fetch_ohlcv_df(ex, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame | None:
    try:
        o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not o or len(o) < 80:
            return None
        df = pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
        return df
    except Exception:
        return None


# =========================
# SCORING (MOMENTUM / 8 GATES)
# =========================
def score_symbol_7gates(df: pd.DataFrame, df_htf: pd.DataFrame) -> dict | None:
    """
    Returns gates(0..7), direction, display score/raw
    Gate8 is reserved for BOTH confirmation outside (cross-exchange).
    """
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    htf_c = df_htf["close"].to_numpy(dtype=float)

    r_tf = rsi(c, 14)
    a_tf = adx(h, l, c, 14)
    mid, up, lowb = bollinger(c, 20, 2.0)
    s20 = sma(c, 20)
    at = atr(h, l, c, 14)

    r_htf = rsi(htf_c, 14)
    htf_sma20 = sma(htf_c, 20)

    if any(np.isnan(arr[-1]) for arr in [r_tf, a_tf, mid, s20, at, r_htf, htf_sma20]):
        return None

    last = float(c[-1])
    s20_last = float(s20[-1])
    mid_last = float(mid[-1])

    # Direction heuristic (momentum follow)
    direction = "LONG" if (last >= s20_last and r_tf[-1] >= 50) else "SHORT"

    gates = 0

    # Gate 1: TF RSI (momentum bias)
    if direction == "LONG" and r_tf[-1] >= RSI_TF_LONG_MIN:
        gates += 1
    if direction == "SHORT" and r_tf[-1] <= RSI_TF_SHORT_MAX:
        gates += 1

    # Gate 2: ADX trend strength
    if a_tf[-1] >= ADX_MIN:
        gates += 1

    # Gate 3: BB mid commitment (avoid mid-noise)
    if direction == "LONG":
        if last >= mid_last * (1 + BB_MID_BUFFER):
            gates += 1
    else:
        if last <= mid_last * (1 - BB_MID_BUFFER):
            gates += 1

    # Gate 4: SMA20 distance (momentum distance)
    sma_dist_pct = (last - s20_last) / s20_last * 100 if s20_last else 0.0
    if direction == "LONG" and sma_dist_pct >= SMA_DIST_PCT_MIN:
        gates += 1
    if direction == "SHORT" and sma_dist_pct <= -SMA_DIST_PCT_MIN:
        gates += 1

    # Gate 5: ATR spike (expansion)
    atr_ema = ema(at, 20)
    atr_spike = (at[-1] / atr_ema[-1]) if atr_ema[-1] and not np.isnan(atr_ema[-1]) else 1.0
    if atr_spike >= ATR_SPIKE_MIN:
        gates += 1

    # Gate 6: HTF trend alignment (price vs HTF SMA20)
    if direction == "LONG" and htf_c[-1] >= htf_sma20[-1]:
        gates += 1
    if direction == "SHORT" and htf_c[-1] <= htf_sma20[-1]:
        gates += 1

    # Gate 7: HTF RSI alignment (real momentum filter)
    if direction == "LONG" and r_htf[-1] >= HTF_RSI_LONG_MIN:
        gates += 1
    if direction == "SHORT" and r_htf[-1] <= HTF_RSI_SHORT_MAX:
        gates += 1

    # Raw score based on 8 gates TOTAL (we’ll add Gate8 later)
    raw_7 = (gates / GATES_TOTAL) * 100.0
    score_7 = step_score(raw_7, SCORE_STEP)

    # Display mapping to keep your old style:
    # LONG: higher is better
    # SHORT: lower is better (invert)
    disp_score = int(score_7) if direction == "LONG" else int(100 - score_7)
    disp_raw = int(round(raw_7)) if direction == "LONG" else int(round(100 - raw_7))

    return {
        "direction": direction,
        "gates_7": int(gates),
        "raw_disp": int(disp_raw),
        "score_disp": int(disp_score),
    }


def strong_flag(direction: str, gates_total: int, score_disp: int) -> bool:
    if gates_total < GATES_REQUIRED_STRONG:
        return False
    if direction == "LONG":
        return score_disp >= STRONG_LONG_SCORE_MIN
    return score_disp <= STRONG_SHORT_SCORE_MAX


# =========================
# UI THEME + TABLE STYLE
# =========================
st.markdown(
    """
<style>
html, body, [class*="css"]  { background-color: #0b0f14 !important; }
.block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }
h1, h2, h3, h4, h5, h6, p, span, div { color: #e6edf3; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


def style_table(df: pd.DataFrame):
    def row_style(row):
        direction = str(row.get("YÖN", ""))
        strong = bool(row.get("STRONG", False))

        # LONG green, SHORT red. STRONG darker.
        if direction == "LONG":
            bg = "#1f6f55" if not strong else "#0f4b37"
        else:
            bg = "#5b1b1b" if not strong else "#3b0f0f"

        return [
            f"background-color: {bg}",
            "color: #e9f1ef",
            "border-color: rgba(255,255,255,0.10)",
            "font-weight: 600" if strong else "font-weight: 500",
        ]

    styler = df.style.apply(lambda r: row_style(r), axis=1)
    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("background-color", "#121826"), ("color", "#dbe7ff"),
                                         ("border-color", "rgba(255,255,255,0.12)"),
                                         ("font-weight", "700")]},
            {"selector": "td", "props": [("border-color", "rgba(255,255,255,0.10)")]},
        ]
    )
    styler = styler.format(
        {
            "FİYAT": "{:,.6f}",
            "QV_24H": "{:,.0f}",
        },
        na_rep="-",
    )
    return styler


# =========================
# TELEGRAM MESSAGE (ONE LIST)
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

    # unified ranking for telegram:
    # - BOTH first
    # - STRONG first
    # - KAPI high
    # - LONG: higher score better
    # - SHORT: lower score better -> convert to rank value
    df["_prio_both"] = (df["SOURCE"] == "BOTH").astype(int)
    df["_prio_strong"] = (df["STRONG"] == True).astype(int)
    df["_score_rank"] = np.where(df["YÖN"] == "SHORT", 100 - df["SKOR"], df["SKOR"])

    df = df.sort_values(
        by=["_prio_both", "_prio_strong", "KAPI", "_score_rank", "QV_24H"],
        ascending=[False, False, False, False, False],
    ).drop(columns=["_prio_both", "_prio_strong", "_score_rank"])

    return df.head(TG_MAX_LIST)


def tg_send_list(title: str, df_list: pd.DataFrame):
    lines = []
    lines.append(f"{title}")
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
# HEADER
# =========================
left, right = st.columns([2, 1], vertical_alignment="center")
with left:
    st.title("🎯 Sniper — Auto (LONG + SHORT)")
    st.caption(
        f"Momentum Mode • TF={TF} • HTF={HTF} • 8/8 Kapı • "
        f"STRONG: LONG ≥ {STRONG_LONG_SCORE_MIN} / SHORT ≤ {STRONG_SHORT_SCORE_MAX} • "
        f"Skor adımı: {SCORE_STEP} • Auto: {AUTO_REFRESH_SEC}s"
    )
with right:
    st.markdown(
        f"""
<div style="text-align:right; padding-top: 6px;">
  <div style="font-size: 12px; opacity: 0.85;">Istanbul Time</div>
  <div style="font-size: 18px; font-weight: 800;">{now_istanbul_str()}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# Auto refresh
st_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")

# Session state
if "sent_keys" not in st.session_state:
    st.session_state.sent_keys = set()
if "last_report_ts" not in st.session_state:
    st.session_state.last_report_ts = 0.0


# =========================
# CONNECT
# =========================
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


# =========================
# SCAN
# =========================
with st.spinner("Marketler taranıyor (hacim/spread filtreleri uygulanıyor)…"):
    ku_syms, ku_tickers = fetch_top_usdt_symbols_and_tickers(kucoin, TOP_N_PER_EXCHANGE)
    ok_syms, ok_tickers = fetch_top_usdt_symbols_and_tickers(okx, TOP_N_PER_EXCHANGE)

ku_set = set(ku_syms)
ok_set = set(ok_syms)

scan_list = sorted(list(ku_set.union(ok_set)))

if not scan_list:
    st.error("Tarama listesi boş. Filtreler çok sert olabilir (hacim/spread).")
    st.stop()

progress = st.progress(0, text=f"Taranıyor: 0/{len(scan_list)}")
rows = []
strong_rows = []

for i, sym in enumerate(scan_list, start=1):
    progress.progress(i / max(1, len(scan_list)), text=f"Taranıyor: {i}/{len(scan_list)} • {sym}")

    base = sym.split("/")[0]

    res_ku = None
    res_ok = None

    # KuCoin calc
    if sym in ku_set:
        df = fetch_ohlcv_df(kucoin, sym, TF, 200)
        dfh = fetch_ohlcv_df(kucoin, sym, HTF, 200)
        if df is not None and dfh is not None:
            res_ku = score_symbol_7gates(df, dfh)

    # OKX calc
    if sym in ok_set:
        df = fetch_ohlcv_df(okx, sym, TF, 200)
        dfh = fetch_ohlcv_df(okx, sym, HTF, 200)
        if df is not None and dfh is not None:
            res_ok = score_symbol_7gates(df, dfh)

    if not res_ku and not res_ok:
        continue

    # Combine logic
    source = None
    chosen = None
    gate8_both_confirm = 0

    if res_ku and res_ok:
        # BOTH symbol exists; Gate8 = 1 only if direction matches
        if res_ku["direction"] == res_ok["direction"]:
            gate8_both_confirm = 1
            direction = res_ku["direction"]
            gates_total = res_ku["gates_7"] + gate8_both_confirm  # gate8 counted
            # Score: average the display scores, then re-step lightly to keep consistency
            score_avg = int(round((res_ku["score_disp"] + res_ok["score_disp"]) / 2))
            raw_avg = int(round((res_ku["raw_disp"] + res_ok["raw_disp"]) / 2))
            chosen = {"direction": direction, "gates_total": int(gates_total), "score": int(score_avg), "raw": int(raw_avg)}
        else:
            # Divergence: still BOTH, but Gate8 = 0 (no confirmation)
            gate8_both_confirm = 0
            # Pick the stricter one by gates_7, then by "conviction"
            pick = res_ku if res_ku["gates_7"] > res_ok["gates_7"] else res_ok
            direction = pick["direction"]
            gates_total = pick["gates_7"] + gate8_both_confirm
            chosen = {"direction": direction, "gates_total": int(gates_total), "score": int(pick["score_disp"]), "raw": int(pick["raw_disp"])}
        source = "BOTH"
    elif res_ku:
        source = "KUCOIN"
        direction = res_ku["direction"]
        chosen = {"direction": direction, "gates_total": int(res_ku["gates_7"]), "score": int(res_ku["score_disp"]), "raw": int(res_ku["raw_disp"])}
    else:
        source = "OKX"
        direction = res_ok["direction"]
        chosen = {"direction": direction, "gates_total": int(res_ok["gates_7"]), "score": int(res_ok["score_disp"]), "raw": int(res_ok["raw_disp"])}

    # Price + QV (fast from cached tickers)
    last_price = 0.0
    qv = 0.0

    if source in ("KUCOIN", "BOTH"):
        t = ku_tickers.get(sym, {})
        last_price = safe_float(t.get("last"), last_price)
        qv = max(qv, safe_float(t.get("quoteVolume"), 0.0))
        if qv <= 0:
            basev = safe_float(t.get("baseVolume"), 0.0)
            if basev > 0 and last_price > 0:
                qv = max(qv, basev * last_price)

    if source in ("OKX", "BOTH"):
        t = ok_tickers.get(sym, {})
        last_price2 = safe_float(t.get("last"), 0.0)
        if last_price2 > 0:
            last_price = max(last_price, last_price2)
        qv2 = safe_float(t.get("quoteVolume"), 0.0)
        if qv2 <= 0:
            basev = safe_float(t.get("baseVolume"), 0.0)
            if basev > 0 and last_price2 > 0:
                qv2 = basev * last_price2
        qv = max(qv, qv2)

    gates_total = int(chosen["gates_total"])
    score = int(chosen["score"])
    raw = int(chosen["raw"])
    direction = chosen["direction"]

    strong = strong_flag(direction, gates_total, score)

    row = {
        "YÖN": direction,
        "COIN": base,
        "SKOR": score,
        "FİYAT": last_price if last_price > 0 else np.nan,
        "RAW": raw,
        "QV_24H": int(qv) if qv else 0,
        "KAPI": gates_total,          # 0..8
        "STRONG": strong,
        "SOURCE": source,             # KUCOIN / OKX / BOTH
    }
    rows.append(row)

    if strong and source == "BOTH":
        strong_rows.append(row)

progress.empty()

df_all = pd.DataFrame(rows)
if df_all.empty:
    st.warning("Hiç aday çıkmadı. Filtreler çok sert olabilir (hacim/spread).")
    st.stop()

# Priority sort:
# 1) BOTH
# 2) STRONG
# 3) KAPI
# 4) best score directionally (SHORT inverted)
df_all["_prio_both"] = (df_all["SOURCE"] == "BOTH").astype(int)
df_all["_prio_strong"] = (df_all["STRONG"] == True).astype(int)
df_all["_score_rank"] = np.where(df_all["YÖN"] == "SHORT", 100 - df_all["SKOR"], df_all["SKOR"])

df_all = df_all.sort_values(
    by=["_prio_both", "_prio_strong", "KAPI", "_score_rank", "QV_24H"],
    ascending=[False, False, False, False, False],
).drop(columns=["_prio_both", "_prio_strong", "_score_rank"])

# Counters (BOTH & STRONG)
strong_long = int(((df_all["STRONG"]) & (df_all["YÖN"] == "LONG") & (df_all["SOURCE"] == "BOTH")).sum())
strong_short = int(((df_all["STRONG"]) & (df_all["YÖN"] == "SHORT") & (df_all["SOURCE"] == "BOTH")).sum())
cnt_long = int((df_all["YÖN"] == "LONG").sum())
cnt_short = int((df_all["YÖN"] == "SHORT").sum())

st.success("Tarama bitti ✅")
st.info(f"✅ STRONG LONG: {strong_long} | 💀 STRONG SHORT: {strong_short} | LONG: {cnt_long} | SHORT: {cnt_short}")

# UI table: 10 long + 10 short (top candidates)
df_long = df_all[df_all["YÖN"] == "LONG"].head(TABLE_N_LONG)
df_short = df_all[df_all["YÖN"] == "SHORT"].head(TABLE_N_SHORT)
df_show = pd.concat([df_long, df_short], ignore_index=True)

if strong_long + strong_short == 0:
    st.warning("⚠️ Şu an STRONG yok. En iyi TOP adaylarla tablo dolduruldu.")
else:
    st.success("✅ STRONG bulundu. Kalan boşluklar TOP adaylarla dolduruldu.")

st.subheader("🎯 SNIPER TABLO")
st.dataframe(style_table(df_show), use_container_width=True, hide_index=True)

# =========================
# TELEGRAM LOGIC
# =========================
# - If NEW strong (BOTH + STRONG + 8/8) appears -> send ONE list message (max 20)
# - If no NEW strong -> every 20 min send report list (best 20 mixed)

if tg_enabled():
    df_strongs_best = _pick_top_for_telegram(df_all, only_strong_both=True)

    # dedupe keys so we don't spam same strongs
    new_found = False
    if not df_strongs_best.empty:
        for _, r in df_strongs_best.iterrows():
            k = f"{r['COIN']}|{r['YÖN']}|{r['SOURCE']}|{r['SKOR']}|{r['KAPI']}"
            if k not in st.session_state.sent_keys:
                st.session_state.sent_keys.add(k)
                new_found = True

    if new_found:
        # One strong list message
        tg_send_list("✅ <b>STRONG SİNYAL (BOTH)</b>", df_strongs_best)
    else:
        # periodic report
        now_ts = time.time()
        if (now_ts - st.session_state.last_report_ts) >= (REPORT_EVERY_MIN * 60):
            df_best20 = _pick_top_for_telegram(df_all, only_strong_both=False)
            tg_send_list("📋 <b>Gözcü Raporu (L/S)</b>", df_best20)
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
