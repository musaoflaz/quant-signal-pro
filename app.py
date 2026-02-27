import os
import time
import html
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import ccxt
import streamlit as st

# ============================================================
# SNIPER v4-dev TEST UI (Streamlit Cloud)
# - No sidebar
# - One-shot scan (RUN SCAN)
# - Includes: BTC Guard + Liquidity Anchor + EMA Retrace Gate + TTL
# ============================================================

# ----------------------------
# GLOBAL CONFIG
# ----------------------------
TF = os.getenv("TF", "15m")
HTF = os.getenv("HTF", "1h")

TOP_N_PER_EXCHANGE = int(os.getenv("TOP_N_PER_EXCHANGE", "60"))
MIN_QV_24H_USDT = float(os.getenv("MIN_QV_24H_USDT", "300000"))
MAX_SPREAD_PCT = float(os.getenv("MAX_SPREAD_PCT", "0.35"))

GATES_TOTAL = int(os.getenv("GATES_TOTAL", "9"))
GATES_REQUIRED_STRONG = int(os.getenv("GATES_REQUIRED_STRONG", "8"))
SCORE_STEP = int(os.getenv("SCORE_STEP", "2"))

# Anti-extreme
RSI_LONG_MAX = float(os.getenv("RSI_LONG_MAX", "75"))
RSI_SHORT_MIN = float(os.getenv("RSI_SHORT_MIN", "25"))

# BTC Guard
BTC_GUARD_ENABLE = os.getenv("BTC_GUARD_ENABLE", "1").strip().lower() in ("1", "true", "yes", "y", "on")
BTC_SYMBOL = os.getenv("BTC_SYMBOL", "BTC/USDT")
BTC_TREND_TF = os.getenv("BTC_TREND_TF", "15m")
BTC_ATR_TF = os.getenv("BTC_ATR_TF", "1h")
BTC_ATR_SPIKE_MULT = float(os.getenv("BTC_ATR_SPIKE_MULT", "2.0"))

# Liquidity Anchor
LQ_ANCHOR_ENABLE = os.getenv("LQ_ANCHOR_ENABLE", "1").strip().lower() in ("1", "true", "yes", "y", "on")
LQ_VWAP_TOL_PCT = float(os.getenv("LQ_VWAP_TOL_PCT", "0.20"))
LQ_EXTEND_MAX_PCT = float(os.getenv("LQ_EXTEND_MAX_PCT", "1.20"))
LQ_POC_BINS = int(os.getenv("LQ_POC_BINS", "24"))
LQ_OB_LOOKBACK = int(os.getenv("LQ_OB_LOOKBACK", "30"))

# EMA Retrace Gate
RETRACE_ENABLE = os.getenv("RETRACE_ENABLE", "1").strip().lower() in ("1", "true", "yes", "y", "on")
RETRACE_EMA_LEN = int(os.getenv("RETRACE_EMA_LEN", "21"))
RETRACE_ZONE_BAND_BPS = int(os.getenv("RETRACE_ZONE_BAND_BPS", "25"))  # 0.25%
RETRACE_NEAR_EMA_BPS = int(os.getenv("RETRACE_NEAR_EMA_BPS", "15"))  # 0.15%
RETRACE_MIN_DISTANCE_FORCE_WAIT_BPS = int(os.getenv("RETRACE_MIN_DISTANCE_FORCE_WAIT_BPS", "60"))  # 0.60%
RETRACE_CONFIRM_CANDLES = int(os.getenv("RETRACE_CONFIRM_CANDLES", "1"))
RETRACE_REQUIRE_REJECTION = os.getenv("RETRACE_REQUIRE_REJECTION", "1").strip().lower() in ("1", "true", "yes", "y", "on")
RETRACE_TTL_MIN = int(os.getenv("RETRACE_TTL_MIN", "45"))
RETRACE_APPLY_MIN_GATES = int(os.getenv("RETRACE_APPLY_MIN_GATES", str(max(1, GATES_REQUIRED_STRONG - 1))))

# Telegram (optional)
TELEGRAM_ENABLE = os.getenv("TELEGRAM_ENABLE", "0").strip().lower() in ("1", "true", "yes", "y", "on")
TG_A_MODE = os.getenv("TG_A_MODE", "1").strip().lower() in ("1", "true", "yes", "y", "on")
DEV_PREFIX = os.getenv("DEV_PREFIX", "[DEV]")


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="SNIPER v4-dev Test UI", layout="wide")
st.title("🔥 SNIPER v4-dev Test UI (Streamlit Cloud)")
st.caption(
    "KuCoin + OKX tarar. BTC Guard + Liquidity Anchor + EMA Retrace Gate + TTL + Near-Strong mantığı test içindir. "
    "Sonsuz worker loop yok: RUN SCAN ile one-shot çalışır."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("TF", TF)
m2.metric("HTF", HTF)
m3.metric("TOP_N/EX", TOP_N_PER_EXCHANGE)
m4.metric("GATES", f"{GATES_REQUIRED_STRONG}/{GATES_TOTAL}")

run_btn = st.button("🚀 RUN SCAN")


# ----------------------------
# Helpers
# ----------------------------
def now_istanbul_str():
    ts = datetime.now(timezone.utc).timestamp() + 3 * 3600
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def step_score(x: float, step: int = SCORE_STEP) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    return int(round(x / step) * step)


def escape_html_text(s: str) -> str:
    return html.escape(str(s), quote=False)


def get_secret(key: str, default: str = "") -> str:
    try:
        if key in st.secrets:
            return str(st.secrets.get(key) or "")
    except Exception:
        pass
    return str(os.getenv(key, default) or "")


# ----------------------------
# Telegram (optional)
# ----------------------------
TG_TOKEN = get_secret("TG_TOKEN", "")
TG_CHAT_ID = get_secret("TG_CHAT_ID", "")


def tg_enabled() -> bool:
    return TELEGRAM_ENABLE and bool(TG_TOKEN) and bool(TG_CHAT_ID)


def _tg_request(payload: dict, timeout_sec: int = 12):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data=payload, timeout=timeout_sec)
        return bool(r.ok), int(getattr(r, "status_code", 0) or 0), (r.text[:300] if hasattr(r, "text") else "")
    except Exception as e:
        return False, 0, str(e)


def send_telegram_html(html_text: str, timeout_sec: int = 12, max_retry: int = 3):
    if not tg_enabled():
        return False, "telegram disabled or missing secrets"

    payload = {
        "chat_id": TG_CHAT_ID,
        "text": html_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    last_err = ""
    for i in range(max(1, max_retry)):
        ok, status, resp = _tg_request(payload, timeout_sec)
        if ok:
            return True, "ok"
        last_err = f"status={status} resp={resp}"
        if status == 429:
            time.sleep(2 + 2 * i)
        else:
            time.sleep(1 + i)
    return False, last_err


def tg_tag(source: str) -> str:
    if source == "BOTH":
        return "✅ <b>VERIFIED</b>"
    if source == "KUCOIN":
        return "🟦 <b>KUCOIN</b>"
    if source == "OKX":
        return "🟨 <b>OKX</b>"
    return f"• <b>{escape_html_text(source)}</b>"


def tg_line(r: dict) -> str:
    direction = str(r.get("YÖN", ""))
    coin = escape_html_text(r.get("COIN", ""))
    score = int(r.get("SKOR", 0))
    gates = int(r.get("KAPI", 0))
    src = str(r.get("SOURCE", ""))
    price = float(r.get("FİYAT", 0.0) or 0.0)
    note = str(r.get("NOTE", "") or "")

    d = "🟢 <b>LONG</b>" if direction == "LONG" else "🔴 <b>SHORT</b>"
    s = (
        f"• <b>{coin}</b>  |  {d}  |  <b>{score}</b>  |  <code>{gates}/{GATES_TOTAL}</code>  |  {tg_tag(src)}\n"
        f"  <i>Fiyat:</i> <code>{price:,.6f}</code>"
    )
    if note:
        s += f"\n  <i>Not:</i> <code>{escape_html_text(note)}</code>"
    return s


def tg_send(title: str, rows: list, note: str = ""):
    title = f"{DEV_PREFIX} {title}".strip()
    lines = [
        f"🎯 <b>{escape_html_text(title)}</b>",
        f"⏱ <code>{escape_html_text(now_istanbul_str())}</code>",
    ]
    if note:
        lines.append(escape_html_text(note))
    lines.append("")

    if not rows:
        lines.append("⚠️ Uygun aday yok.")
    else:
        for r in rows[:20]:
            lines.append(tg_line(r))
            lines.append("")

    return send_telegram_html("\n".join(lines).strip())


# ----------------------------
# Indicators
# ----------------------------
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

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out = dx.ewm(alpha=1 / period, adjust=False).mean()
    return out.to_numpy()


# ----------------------------
# Retrace Gate
# ----------------------------
@dataclass
class RetraceWaitState:
    side: str
    created_ts: float
    expires_ts: float
    ema_at_creation: float
    zone_low: float
    zone_high: float


class RetraceGate:
    def __init__(self):
        self.wait: Dict[str, RetraceWaitState] = {}

    @staticmethod
    def _ema_last(df: pd.DataFrame, length: int) -> Optional[float]:
        try:
            if df is None or df.empty or "close" not in df.columns:
                return None
            if len(df) < max(30, length + 5):
                return None
            close = pd.to_numeric(df["close"], errors="coerce")
            e = close.ewm(span=length, adjust=False).mean().iloc[-1]
            if pd.isna(e):
                return None
            return float(e)
        except Exception:
            return None

    @staticmethod
    def _zone(ema_val: float, band_bps: int) -> Tuple[float, float]:
        band = (band_bps / 10000.0) * ema_val
        return ema_val - band, ema_val + band

    @staticmethod
    def _dist_bps(price: float, ema_val: float) -> float:
        if not ema_val:
            return 0.0
        return ((price - ema_val) / ema_val) * 10000.0

    def evaluate(self, symbol: str, side: str, df_15m: pd.DataFrame) -> Tuple[bool, str]:
        if not RETRACE_ENABLE:
            return False, ""

        side = (side or "").upper().strip()
        if side not in ("LONG", "SHORT"):
            return False, ""

        ema_val = self._ema_last(df_15m, RETRACE_EMA_LEN)
        if ema_val is None:
            return False, ""

        if df_15m is None or df_15m.empty:
            return False, ""

        for col in ("open", "high", "low", "close"):
            if col not in df_15m.columns:
                return False, ""

        price = float(df_15m["close"].iloc[-1])
        zone_low, zone_high = self._zone(ema_val, RETRACE_ZONE_BAND_BPS)
        dist_bps = self._dist_bps(price, ema_val)
        now_ts = time.time()

        def confirm_trigger() -> bool:
            n = int(max(1, RETRACE_CONFIRM_CANDLES))
            if len(df_15m) < n:
                return False
            tail = df_15m.tail(n)
            in_zone = (tail["low"] <= zone_high) & (tail["high"] >= zone_low)

            if side == "LONG":
                close_ok = tail["close"] >= ema_val
                if not RETRACE_REQUIRE_REJECTION:
                    return bool((in_zone & close_ok).all())
                wick_ok = (tail["close"] - tail["low"]) >= (tail["high"] - tail["close"]) * 0.6
                return bool((in_zone & close_ok & wick_ok).all())

            close_ok = tail["close"] <= ema_val
            if not RETRACE_REQUIRE_REJECTION:
                return bool((in_zone & close_ok).all())
            wick_ok = (tail["high"] - tail["close"]) >= (tail["close"] - tail["low"]) * 0.6
            return bool((in_zone & close_ok & wick_ok).all())

        # already waiting
        if symbol in self.wait:
            stt = self.wait[symbol]
            if stt.side != side:
                self.wait.pop(symbol, None)
            else:
                if now_ts >= stt.expires_ts:
                    self.wait.pop(symbol, None)
                    return False, f"RETRACE:EXPIRED(ttl_min={RETRACE_TTL_MIN})"

                if confirm_trigger():
                    self.wait.pop(symbol, None)
                    return False, f"ENTRY_TRIGGERED_{side}(ema{RETRACE_EMA_LEN}={ema_val:.6f},zone=[{zone_low:.6f}-{zone_high:.6f}])"

                ttl_left = max(0, int((stt.expires_ts - now_ts) // 60))
                return True, (
                    f"WAIT:EMA_RETRACE_{side}(ema{RETRACE_EMA_LEN}={ema_val:.6f},"
                    f"zone=[{zone_low:.6f}-{zone_high:.6f}],price={price:.6f},"
                    f"dist_bps={dist_bps:.1f},ttl_left_min={ttl_left})"
                )

        # near EMA => no forced wait
        if abs(dist_bps) <= RETRACE_NEAR_EMA_BPS:
            if confirm_trigger():
                return False, f"ENTRY_TRIGGERED_{side}(ema{RETRACE_EMA_LEN}={ema_val:.6f},zone=[{zone_low:.6f}-{zone_high:.6f}])"
            return False, ""

        # not far enough => no forced wait
        if abs(dist_bps) < RETRACE_MIN_DISTANCE_FORCE_WAIT_BPS:
            return False, ""

        # create wait
        self.wait[symbol] = RetraceWaitState(
            side=side,
            created_ts=now_ts,
            expires_ts=now_ts + RETRACE_TTL_MIN * 60,
            ema_at_creation=float(ema_val),
            zone_low=float(zone_low),
            zone_high=float(zone_high),
        )
        return True, (
            f"WAIT:EMA_RETRACE_{side}(ema{RETRACE_EMA_LEN}={ema_val:.6f},"
            f"zone=[{zone_low:.6f}-{zone_high:.6f}],price={price:.6f},"
            f"dist_bps={dist_bps:.1f},ttl_left_min={RETRACE_TTL_MIN})"
        )


RETRACE_GATE = RetraceGate()


# ----------------------------
# Liquidity Anchor
# ----------------------------
def vwap_from_df(df: pd.DataFrame) -> float:
    try:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].replace(0, np.nan)
        num = (tp * vol).sum()
        den = vol.sum()
        if den and not np.isnan(den) and den > 0:
            return float(num / den)
    except Exception:
        pass
    return float("nan")


def poc_from_df(df: pd.DataFrame, bins: int = 24) -> float:
    try:
        close = df["close"].to_numpy(dtype=float)
        vol = df["volume"].to_numpy(dtype=float)
        if len(close) < 80:
            return float("nan")
        lo = np.nanmin(close)
        hi = np.nanmax(close)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return float("nan")
        edges = np.linspace(lo, hi, bins + 1)
        idx = np.clip(np.digitize(close, edges) - 1, 0, bins - 1)
        agg = np.zeros(bins, dtype=float)
        for i, b in enumerate(idx):
            vv = vol[i] if np.isfinite(vol[i]) else 0.0
            agg[b] += vv
        bmax = int(np.argmax(agg))
        return float((edges[bmax] + edges[bmax + 1]) / 2.0)
    except Exception:
        return float("nan")


def detect_orderblock_simple(df: pd.DataFrame, direction: str, lookback: int = 30) -> bool:
    try:
        if df is None or df.empty:
            return False
        d = df.tail(max(lookback + 6, 40)).copy()
        o = d["open"].to_numpy(dtype=float)
        h = d["high"].to_numpy(dtype=float)
        l = d["low"].to_numpy(dtype=float)
        c = d["close"].to_numpy(dtype=float)
        n = len(d)
        if n < 20:
            return False

        rng = np.maximum(h - l, 1e-12)
        body = np.abs(c - o)
        body_ratio = body / rng
        last = float(c[-1])

        if direction == "LONG":
            for i in range(n - 6, 5, -1):
                if c[i] < o[i] and body_ratio[i] >= 0.60:
                    hi = float(h[i])
                    if (c[i + 1] > hi * 1.003) or (c[i + 2] > hi * 1.003) or (c[i + 3] > hi * 1.003):
                        return last >= hi
            return False

        for i in range(n - 6, 5, -1):
            if c[i] > o[i] and body_ratio[i] >= 0.60:
                lo = float(l[i])
                if (c[i + 1] < lo * 0.997) or (c[i + 2] < lo * 0.997) or (c[i + 3] < lo * 0.997):
                    return last <= lo
        return False
    except Exception:
        return False


def liquidity_anchor_ok(df: pd.DataFrame, direction: str) -> Tuple[bool, str]:
    if not LQ_ANCHOR_ENABLE:
        return True, ""

    try:
        if df is None or df.empty:
            return False, "LQ:NoData"

        last = float(df["close"].iloc[-1])
        vwap = vwap_from_df(df)
        poc = poc_from_df(df, bins=LQ_POC_BINS)
        ob_ok = detect_orderblock_simple(df, direction, lookback=LQ_OB_LOOKBACK)

        parts = []

        if np.isfinite(vwap) and vwap > 0:
            tol = LQ_VWAP_TOL_PCT / 100.0
            ext = (abs(last - vwap) / vwap) * 100.0
            if ext > LQ_EXTEND_MAX_PCT:
                parts.append("VWAP-EXT")
                vwap_ok = False
            else:
                if direction == "LONG":
                    vwap_ok = (last >= vwap * (1 - tol))
                else:
                    vwap_ok = (last <= vwap * (1 + tol))
                parts.append("VWAP+" if vwap_ok else "VWAP-")
        else:
            vwap_ok = False
            parts.append("VWAP?")

        if np.isfinite(poc) and poc > 0:
            dist_poc = (abs(last - poc) / poc) * 100.0
            poc_ok = dist_poc <= (LQ_EXTEND_MAX_PCT * 1.2)
            parts.append("POC+" if poc_ok else "POC-")
        else:
            poc_ok = False
            parts.append("POC?")

        parts.append("OB+" if ob_ok else "OB-")

        ok = bool(vwap_ok and poc_ok and ob_ok)
        return ok, "LQ:" + ",".join(parts)
    except Exception:
        return False, "LQ:ERR"


# ----------------------------
# Exchange helpers
# ----------------------------
def get_exchanges():
    kucoin = ccxt.kucoin({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "spot"}})
    okx = ccxt.okx({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "spot"}})
    return kucoin, okx


def symbol_clean(sym: str) -> str:
    return sym.replace(":USDT", "")


def is_usdt_spot_symbol(sym: str) -> bool:
    s = symbol_clean(sym)
    return s.endswith("/USDT") and (":" not in s)


def is_junk_symbol(base: str) -> bool:
    b = base.upper()
    junk = {"USDT", "USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "EUR", "EURT", "3S", "3L", "5S", "5L", "BULL", "BEAR", "UP", "DOWN"}
    if b in junk:
        return True
    for suf in ["3L", "3S", "5L", "5S", "BULL", "BEAR", "UP", "DOWN"]:
        if b.endswith(suf):
            return True
    return False


def load_markets_safe(ex, name: str) -> bool:
    try:
        ex.load_markets()
        return True
    except Exception as e:
        st.error(f"{name}: load_markets failed: {e}")
        return False


def fetch_top_usdt_symbols(ex, top_n: int) -> List[str]:
    try:
        tickers = ex.fetch_tickers()
    except Exception as e:
        st.warning(f"fetch_tickers failed: {e}")
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


def fetch_ohlcv_df(ex, symbol: str, timeframe: str, limit: int = 200):
    try:
        o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not o or len(o) < 120:
            return None
        return pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
    except Exception:
        return None


# ----------------------------
# BTC context
# ----------------------------
def get_btc_context(kucoin, okx) -> dict:
    ctx = {
        "ok": False,
        "symbol": BTC_SYMBOL,
        "trend_tf": BTC_TREND_TF,
        "atr_tf": BTC_ATR_TF,
        "below_ema21_15m": None,
        "last": None,
        "ema21": None,
        "atr": None,
        "range": None,
        "atr_shock": None,
        "src": None,
    }
    if not BTC_GUARD_ENABLE:
        return ctx

    for ex, name in [(kucoin, "KUCOIN"), (okx, "OKX")]:
        try:
            df_tr = fetch_ohlcv_df(ex, BTC_SYMBOL, BTC_TREND_TF, 220)
            df_at = fetch_ohlcv_df(ex, BTC_SYMBOL, BTC_ATR_TF, 220)
            if df_tr is None or df_at is None or df_tr.empty or df_at.empty:
                continue

            c = df_tr["close"].to_numpy(dtype=float)
            e21 = ema(c, 21)
            if np.isnan(e21[-1]) or np.isnan(c[-1]):
                continue

            last = float(c[-1])
            ema21v = float(e21[-1])
            below = bool(last < ema21v)

            ah = df_at["high"].to_numpy(dtype=float)
            al = df_at["low"].to_numpy(dtype=float)
            ac = df_at["close"].to_numpy(dtype=float)
            a = atr(ah, al, ac, 14)

            if np.isnan(a[-1]) or np.isnan(ah[-1]) or np.isnan(al[-1]):
                atrv, rngv, shock = None, None, None
            else:
                atrv = float(a[-1])
                rngv = float(ah[-1] - al[-1])
                shock = bool(rngv > (BTC_ATR_SPIKE_MULT * atrv)) if (atrv and atrv > 0) else False

            ctx.update({
                "ok": True,
                "below_ema21_15m": below,
                "last": last,
                "ema21": ema21v,
                "atr": atrv,
                "range": rngv,
                "atr_shock": shock,
                "src": name,
            })
            return ctx
        except Exception:
            continue

    return ctx


# ----------------------------
# Scoring
# ----------------------------
def score_symbol(df: pd.DataFrame, df_htf: pd.DataFrame, btc_ctx: dict):
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    htf_h = df_htf["high"].to_numpy(dtype=float)
    htf_l = df_htf["low"].to_numpy(dtype=float)
    htf_c = df_htf["close"].to_numpy(dtype=float)

    r = rsi(c, 14)
    a = adx(h, l, c, 14)
    mid, _, _ = bollinger(c, 20, 2.0)
    s20 = sma(c, 20)
    at = atr(h, l, c, 14)

    htf_s20 = sma(htf_c, 20)
    htf_adx = adx(htf_h, htf_l, htf_c, 14)

    if any(np.isnan(x[-1]) for x in [r, a, mid, s20, at, htf_s20, htf_adx]):
        return None

    last = float(c[-1])
    direction = "LONG" if (last >= s20[-1] and r[-1] >= 50) else "SHORT"
    gates = 0

    # Anti-extreme
    if direction == "LONG" and float(r[-1]) > RSI_LONG_MAX:
        return {"direction": direction, "gates": 0, "raw": 0, "score": 0, "block_note": f"WAIT:RSI>{RSI_LONG_MAX:.0f}"}
    if direction == "SHORT" and float(r[-1]) < RSI_SHORT_MIN:
        return {"direction": direction, "gates": 0, "raw": 0, "score": 0, "block_note": f"WAIT:RSI<{RSI_SHORT_MIN:.0f}"}

    # 1 RSI bias
    if direction == "LONG" and r[-1] >= 58:
        gates += 1
    if direction == "SHORT" and r[-1] <= 42:
        gates += 1

    # 2 ADX
    if a[-1] >= 22:
        gates += 1

    # 3 ATR spike
    atr_ema = ema(at, 20)
    atr_spike = (at[-1] / atr_ema[-1]) if (atr_ema[-1] and not np.isnan(atr_ema[-1])) else 1.0
    if atr_spike >= 1.08:
        gates += 1

    # 4 SMA distance
    sma_dist_pct = (last - s20[-1]) / s20[-1] * 100 if s20[-1] else 0.0
    if direction == "LONG" and sma_dist_pct >= 0.20:
        gates += 1
    if direction == "SHORT" and sma_dist_pct <= -0.20:
        gates += 1

    # 5 HTF trend
    if direction == "LONG" and htf_c[-1] >= htf_s20[-1]:
        gates += 1
    if direction == "SHORT" and htf_c[-1] <= htf_s20[-1]:
        gates += 1

    # 6 BB mid trend
    if direction == "LONG" and last >= mid[-1] * 1.001:
        gates += 1
    if direction == "SHORT" and last <= mid[-1] * 0.999:
        gates += 1

    # 7 HTF ADX
    if htf_adx[-1] >= 20:
        gates += 1

    # 8 Breakout prev3
    if len(h) >= 5:
        prev3_high = float(np.max(h[-4:-1]))
        prev3_low = float(np.min(l[-4:-1]))
        if direction == "LONG" and last > prev3_high:
            gates += 1
        if direction == "SHORT" and last < prev3_low:
            gates += 1

    # 9 BTC EMA21 gate
    if BTC_GUARD_ENABLE and btc_ctx and btc_ctx.get("ok") and (btc_ctx.get("below_ema21_15m") is not None):
        btc_below = bool(btc_ctx["below_ema21_15m"])
        if direction == "LONG" and (btc_below is False):
            gates += 1
        if direction == "SHORT" and (btc_below is True):
            gates += 1

    raw = (gates / max(1, GATES_TOTAL)) * 100.0
    score = step_score(raw, SCORE_STEP)

    disp_score = score if direction == "LONG" else (100 - score)
    disp_raw = int(round(raw)) if direction == "LONG" else int(round(100 - raw))

    return {"direction": direction, "gates": int(gates), "raw": int(disp_raw), "score": int(disp_score), "block_note": ""}


def strong_flag(gates: int) -> bool:
    return int(gates) >= int(GATES_REQUIRED_STRONG)


# ----------------------------
# RUN SCAN (one-shot)
# ----------------------------
def run_scan() -> Tuple[pd.DataFrame, dict]:
    kucoin, okx = get_exchanges()
    if not (load_markets_safe(kucoin, "KuCoin") and load_markets_safe(okx, "OKX")):
        raise RuntimeError("load_markets failed")

    btc_ctx = get_btc_context(kucoin, okx)

    ku_syms = fetch_top_usdt_symbols(kucoin, TOP_N_PER_EXCHANGE)
    ok_syms = fetch_top_usdt_symbols(okx, TOP_N_PER_EXCHANGE)

    ku_set = set(ku_syms)
    ok_set = set(ok_syms)
    scan_list = sorted(list(ku_set.union(ok_set)))

    rows = []
    for sym in scan_list:
        base = sym.split("/")[0]

        res_ku = None
        res_ok = None

        df_for_lq = None
        df_for_retrace = None

        if sym in ku_set:
            df = fetch_ohlcv_df(kucoin, sym, TF, 200)
            dfh = fetch_ohlcv_df(kucoin, sym, HTF, 200)
            if df is not None and dfh is not None:
                res_ku = score_symbol(df, dfh, btc_ctx)
                if df_for_lq is None:
                    df_for_lq = df
                if df_for_retrace is None:
                    df_for_retrace = df

        if sym in ok_set:
            df2 = fetch_ohlcv_df(okx, sym, TF, 200)
            dfh2 = fetch_ohlcv_df(okx, sym, HTF, 200)
            if df2 is not None and dfh2 is not None:
                res_ok = score_symbol(df2, dfh2, btc_ctx)
                if df_for_lq is None:
                    df_for_lq = df2
                if df_for_retrace is None:
                    df_for_retrace = df2

        if not (res_ku or res_ok):
            continue

        source = "KUCOIN" if (res_ku and not res_ok) else "OKX" if (res_ok and not res_ku) else "BOTH"

        if res_ku and res_ok:
            same_dir = (res_ku.get("direction") == res_ok.get("direction"))
            direction = res_ku.get("direction")
            gates = min(int(res_ku.get("gates", 0)), int(res_ok.get("gates", 0)))
            score = int(round((float(res_ku.get("score", 0)) + float(res_ok.get("score", 0))) / 2.0))
            raw = int(round((float(res_ku.get("raw", 0)) + float(res_ok.get("raw", 0))) / 2.0))
            block_note = (res_ku.get("block_note") or "") or (res_ok.get("block_note") or "")
        else:
            pick = res_ku if res_ku else res_ok
            same_dir = False
            direction = pick.get("direction")
            gates = int(pick.get("gates", 0))
            score = int(pick.get("score", 0))
            raw = int(pick.get("raw", 0))
            block_note = pick.get("block_note", "")

        if not direction:
            continue

        # price + qv
        last_price = 0.0
        qv = 0.0
        try:
            if res_ku:
                t = kucoin.fetch_ticker(sym)
                last_price = max(last_price, safe_float(t.get("last"), 0.0))
                qv = max(qv, safe_float(t.get("quoteVolume"), 0.0))
            if res_ok:
                t2 = okx.fetch_ticker(sym)
                last_price = max(last_price, safe_float(t2.get("last"), 0.0))
                qv = max(qv, safe_float(t2.get("quoteVolume"), 0.0))
        except Exception:
            pass

        strong = strong_flag(int(gates))
        verified = bool(res_ku and res_ok and same_dir and strong and int(gates) >= int(GATES_REQUIRED_STRONG))

        status = "OK"
        note_parts = []

        if block_note:
            status = "WAIT"
            note_parts.append(block_note)

        # BTC guard
        if BTC_GUARD_ENABLE and btc_ctx.get("ok"):
            if direction == "LONG" and btc_ctx.get("below_ema21_15m") is True:
                status = "WAIT"
                note_parts.append("WAIT:MarketRisk(BTC<EMA21)")
            if btc_ctx.get("atr_shock") is True:
                status = "WAIT"
                note_parts.append("WAIT:BTC_ATR_SHOCK")

        # Liquidity anchor blocks only strong/verified
        lq_ok, lq_note = liquidity_anchor_ok(df_for_lq, direction)
        if lq_note:
            note_parts.append(lq_note)
        if strong and (not lq_ok):
            status = "WAIT"
            strong = False
            verified = False
            note_parts.append("WAIT:LowLiquidity")

        # Retrace gate
        if (
            RETRACE_ENABLE
            and status == "OK"
            and int(gates) >= int(RETRACE_APPLY_MIN_GATES)
            and df_for_retrace is not None
        ):
            is_wait, rnote = RETRACE_GATE.evaluate(sym, direction, df_for_retrace)
            if rnote:
                note_parts.append(rnote)
            if is_wait:
                status = "WAIT"
                strong = False
                verified = False

        note = " | ".join([p for p in note_parts if p]).strip()

        rows.append({
            "YÖN": direction,
            "COIN": base,
            "SKOR": int(score),
            "FİYAT": float(last_price),
            "RAW": int(raw),
            "QV_24H": int(qv) if qv else 0,
            "KAPI": int(gates),
            "STRONG": bool(strong),
            "VERIFIED": bool(verified),
            "SOURCE": "BOTH" if (res_ku and res_ok and same_dir) else source,
            "STATUS": status,
            "NOTE": note,
        })

    df_all = pd.DataFrame(rows)
    meta = {"btc": btc_ctx, "scan_count": len(scan_list), "now": now_istanbul_str()}
    return df_all, meta


# ----------------------------
# Execute
# ----------------------------
if run_btn:
    with st.spinner("Veri çekiliyor... KuCoin + OKX taranıyor..."):
        t0 = time.time()
        try:
            df_all, meta = run_scan()
            dt = time.time() - t0

            st.success(f"Scan OK | symbols_scanned={meta['scan_count']} | rows={len(df_all)} | time={dt:.1f}s | {meta['now']}")

            with st.expander("BTC Guard Context"):
                st.json(meta.get("btc") or {})

            if df_all.empty:
                st.warning("Aday yok.")
            else:
                df_all["_prio_verified"] = (df_all["VERIFIED"] == True).astype(int)
                df_all["_prio_strong"] = (df_all["STRONG"] == True).astype(int)
                df_all["_score_rank"] = np.where(df_all["YÖN"] == "SHORT", 100 - df_all["SKOR"], df_all["SKOR"])
                df_all = df_all.sort_values(
                    by=["_prio_verified", "_prio_strong", "KAPI", "_score_rank", "QV_24H"],
                    ascending=[False, False, False, False, False]
                ).drop(columns=["_prio_verified", "_prio_strong", "_score_rank"])

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("TOTAL", int(len(df_all)))
                c2.metric("STRONG", int((df_all["STRONG"] == True).sum()))
                c3.metric("VERIFIED", int((df_all["VERIFIED"] == True).sum()))
                c4.metric("WAIT", int((df_all["STATUS"] == "WAIT").sum()))

                st.subheader("Top 50")
                st.dataframe(df_all.head(50), use_container_width=True)

                st.subheader("WAIT (Filters / Retrace)")
                st.dataframe(df_all[df_all["STATUS"] == "WAIT"].head(200), use_container_width=True)

                st.subheader("Only STRONG")
                st.dataframe(df_all[df_all["STRONG"] == True].head(200), use_container_width=True)

                # Telegram (optional)
                if tg_enabled():
                    verified_rows = df_all[df_all["VERIFIED"] == True].to_dict("records")
                    best_rows = df_all.head(50).to_dict("records")

                    if TG_A_MODE:
                        if verified_rows:
                            ok, info = tg_send("VERIFIED STRONG (BOTH)", verified_rows[:20], note="🔒 KuCoin + OKX aynı yön + STRONG + Filtreler OK")
                        else:
                            ok, info = tg_send("Gözcü Raporu (En Olası Adaylar)", best_rows[:20], note="📌 Verified STRONG yok. Liste 'en olası' adaylardır.")
                    else:
                        ok, info = tg_send("Top Adaylar", best_rows[:20], note="(A MODE kapalı)")

                    st.info(f"Telegram: {'OK' if ok else 'FAIL'} | {info}")
                else:
                    st.caption("Telegram kapalı (Secrets yoksa normal).")

        except Exception as e:
            st.error(f"Scan ERROR: {e}")
else:
    st.caption("Hazır. RUN SCAN'e basınca tablo dolacak.")
