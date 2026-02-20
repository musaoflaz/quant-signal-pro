from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from urllib import request, parse

import numpy as np
import pandas as pd
import streamlit as st
import ccxt


# =========================================================
# ✅ TELEGRAM (buraya yapıştır)
# =========================================================
TELEGRAM_BOT_TOKEN = "8330775219:AAHx20fZA6C3ONs5S8ELQrMpFEYba-bPN1k"
TELEGRAM_CHAT_ID = "1358384022"

# =========================================================
# FINAL BASE (KUCOIN + OKX) — Auto + Telegram
# - Dark theme
# - 6 Kapı (TF 15m + HTF 1h)
# - STRONG: KAPI==6 AND (SKOR>=90 LONG / SKOR<=10 SHORT)
# - Score step: 5
# - Table: 20 rows (BOTH first)
# - Telegram:
#   * STRONG gelince anında mesaj (cooldown)
#   * 20 dakikada bir rapor (STRONG yoksa bile)
# =========================================================

IST = ZoneInfo("Europe/Istanbul")

TF = "15m"
HTF = "1h"

AUTO_REFRESH_SEC = 60           # sayfa yenileme
REPORT_EVERY_MIN = 20           # 20 dk rapor
ALERT_COOLDOWN_MIN = 30         # aynı sinyali spam yapmasın

TOP_PER_EX = 200                # kalite için 200
TABLE_ROWS = 20
FALLBACK_LONG = 10
FALLBACK_SHORT = 10

CANDLE_LIMIT = 220
SMA_PERIOD = 20
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
ADX_PERIOD = 14

SCORE_STEP = 5
STRONG_LONG_MIN = 90
STRONG_SHORT_MAX = 10

# kalite filtresi (çöp coin azaltır)
MIN_QV_24H_USDT = 1_000_000

STABLE_BASES = {
    "USDT", "USDC", "DAI", "TUSD", "USDP", "FDUSD", "BUSD", "PYUSD",
    "EURC", "USDD", "USDG", "USDE", "FRAX", "LUSD", "SUSD", "USTC",
}

ADX_MIN_TF = 18.0
ADX_MIN_HTF = 15.0


# -----------------------------
# Telegram (stdlib only)
# -----------------------------
def telegram_send(token: str, chat_id: str, text: str, timeout: int = 12) -> tuple[bool, str]:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        form = parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
        req = request.Request(
            url=url,
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "sniper-pro"},
            method="POST",
        )
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        return True, body[:200]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def now_ist() -> datetime:
    return datetime.now(IST)


def can_send_report() -> bool:
    last = st.session_state.get("tg_last_report")
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except Exception:
        return True
    return now_ist() - last_dt >= timedelta(minutes=REPORT_EVERY_MIN)


def mark_report_sent():
    st.session_state["tg_last_report"] = now_ist().isoformat()


def can_send_alert(key: str) -> bool:
    # key like "BTC|STRONG_LONG"
    last_map = st.session_state.get("tg_last_alert_map", {})
    last = last_map.get(key)
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except Exception:
        return True
    return now_ist() - last_dt >= timedelta(minutes=ALERT_COOLDOWN_MIN)


def mark_alert_sent(key: str):
    last_map = st.session_state.get("tg_last_alert_map", {})
    last_map[key] = now_ist().isoformat()
    st.session_state["tg_last_alert_map"] = last_map


def telegram_enabled() -> bool:
    return (
        isinstance(TELEGRAM_BOT_TOKEN, str) and TELEGRAM_BOT_TOKEN.strip() and
        isinstance(TELEGRAM_CHAT_ID, str) and TELEGRAM_CHAT_ID.strip() and
        "PASTE_" not in TELEGRAM_BOT_TOKEN and "PASTE_" not in TELEGRAM_CHAT_ID
    )


# -----------------------------
# Indicators (pure pandas/numpy)
# -----------------------------
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def bollinger(series: pd.Series, period: int, n_std: float):
    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + (n_std * std)
    lower = mid - (n_std * std)
    return mid, upper, lower


def rsi_wilder(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0.0, np.nan)

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx_v = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx_v.fillna(0.0)


def round_step(x: float, step: int) -> int:
    return int(step * round(float(x) / step))


# -----------------------------
# Scoring + 6 gates
# -----------------------------
def raw_score_core(close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float) -> float:
    score = 50.0
    score += 20.0 if close > sma20_v else -20.0
    if rsi_v < 35.0:
        score += 40.0
    elif rsi_v > 65.0:
        score -= 40.0
    if close <= bb_low:
        score += 40.0
    elif close >= bb_up:
        score -= 40.0
    return float(np.clip(score, 0.0, 100.0))


def direction_from_raw(raw: float) -> str:
    return "LONG" if raw >= 50.0 else "SHORT"


def gates_6(
    direction: str,
    close: float, sma20_v: float, rsi_v: float, bb_low: float, bb_up: float, adx_v: float,
    h_close: float | None, h_sma20: float | None, h_rsi: float | None, h_adx: float | None
) -> int:
    g = 0

    # 1) TF trend
    if direction == "LONG" and close > sma20_v: g += 1
    if direction == "SHORT" and close < sma20_v: g += 1

    # 2) TF RSI sanity
    if direction == "LONG" and rsi_v <= 55.0: g += 1
    if direction == "SHORT" and rsi_v >= 45.0: g += 1

    # 3) TF BB edge touch
    if direction == "LONG" and close <= bb_low: g += 1
    if direction == "SHORT" and close >= bb_up: g += 1

    # 4) TF ADX
    if adx_v >= ADX_MIN_TF: g += 1

    # 5) HTF trend
    if h_close is not None and h_sma20 is not None:
        if direction == "LONG" and h_close > h_sma20: g += 1
        if direction == "SHORT" and h_close < h_sma20: g += 1

    # 6) HTF momentum + HTF ADX
    ok_momo = False
    if h_rsi is not None:
        if direction == "LONG" and h_rsi <= 60.0: ok_momo = True
        if direction == "SHORT" and h_rsi >= 40.0: ok_momo = True

    ok_adx = False
    if h_adx is not None and h_adx >= ADX_MIN_HTF:
        ok_adx = True

    if ok_momo and ok_adx:
        g += 1

    return g


def is_strong(direction: str, skor: int, kapi: int) -> bool:
    if kapi != 6:
        return False
    if direction == "LONG":
        return skor >= STRONG_LONG_MIN
    return skor <= STRONG_SHORT_MAX


# -----------------------------
# CCXT helpers
# -----------------------------
def make_exchange(name: str) -> ccxt.Exchange:
    params = {"enableRateLimit": True, "timeout": 20000}
    if name == "kucoin":
        return ccxt.kucoin(params)
    if name == "okx":
        return ccxt.okx(params)
    raise ValueError("unsupported exchange")


def safe_load_markets(ex: ccxt.Exchange) -> dict:
    try:
        return ex.load_markets()
    except Exception:
        return {}


def safe_fetch_tickers(ex: ccxt.Exchange, symbols: list[str]) -> dict:
    try:
        return ex.fetch_tickers(symbols)
    except Exception:
        try:
            all_t = ex.fetch_tickers()
            return {s: all_t.get(s) for s in symbols if s in all_t}
        except Exception:
            return {}


def compute_qv_24h(ticker: dict) -> float:
    if not ticker or not isinstance(ticker, dict):
        return 0.0
    qv = ticker.get("quoteVolume")
    if qv is not None:
        try:
            return float(qv)
        except Exception:
            pass
    bv = ticker.get("baseVolume")
    last = ticker.get("last")
    try:
        if bv is not None and last is not None:
            return float(bv) * float(last)
    except Exception:
        pass
    return 0.0


def fetch_ohlcv_df(ex: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    try:
        o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not o or len(o) < 60:
            return None
        return pd.DataFrame(o, columns=["ts", "open", "high", "low", "close", "volume"])
    except Exception:
        return None


def get_usdt_spot_symbols(markets: dict) -> list[str]:
    syms = []
    for sym, m in markets.items():
        if not m or not isinstance(m, dict):
            continue
        if not m.get("active", True):
            continue
        if not m.get("spot", False):
            continue
        if m.get("quote") != "USDT":
            continue
        base = m.get("base") or (sym.split("/")[0] if "/" in sym else "")
        if str(base).upper() in STABLE_BASES:
            continue
        syms.append(sym)
    return sorted(set(syms))


def scan_exchange(exchange_name: str) -> tuple[pd.DataFrame, dict]:
    ex = make_exchange(exchange_name)
    markets = safe_load_markets(ex)
    syms = get_usdt_spot_symbols(markets)

    tickers = safe_fetch_tickers(ex, syms)

    ranked = []
    for s in syms:
        qv = compute_qv_24h(tickers.get(s) or {})
        if qv >= MIN_QV_USDT_24H:
            ranked.append((s, qv))

    ranked.sort(key=lambda x: x[1], reverse=True)
    top_syms = [s for s, _ in ranked[:TOP_PER_EX]]

    meta = {"universe": len(syms), "after_liq": len(ranked), "scanned": len(top_syms)}

    rows = []
    for sym in top_syms:
        df = fetch_ohlcv_df(ex, sym, TF, CANDLE_LIMIT)
        if df is None:
            continue

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        sma20_s = sma(close, SMA_PERIOD)
        _, bb_up_s, bb_low_s = bollinger(close, BB_PERIOD, BB_STD)
        rsi_s = rsi_wilder(close, RSI_PERIOD)
        adx_s = adx(high, low, close, ADX_PERIOD)

        last_close = float(close.iloc[-1])
        last_sma20 = float(sma20_s.iloc[-1])
        last_rsi = float(rsi_s.iloc[-1])
        last_low = float(bb_low_s.iloc[-1])
        last_up = float(bb_up_s.iloc[-1])
        last_adx = float(adx_s.iloc[-1])

        if any(np.isnan([last_sma20, last_rsi, last_low, last_up, last_adx])):
            continue

        raw = raw_score_core(last_close, last_sma20, last_rsi, last_low, last_up)
        direction = direction_from_raw(raw)
        skor = round_step(raw, SCORE_STEP)

        # HTF ALWAYS (for KAPI to reach 6)
        h_close = h_sma20 = h_rsi = h_adx = None
        dfh = fetch_ohlcv_df(ex, sym, HTF, CANDLE_LIMIT)
        if dfh is not None:
            hc = dfh["close"].astype(float)
            hh = dfh["high"].astype(float)
            hl = dfh["low"].astype(float)
            hsma20 = sma(hc, SMA_PERIOD)
            hrsi = rsi_wilder(hc, RSI_PERIOD)
            hadx = adx(hh, hl, hc, ADX_PERIOD)
            try:
                h_close = float(hc.iloc[-1])
                h_sma20 = float(hsma20.iloc[-1])
                hv_rsi = float(hrsi.iloc[-1])
                hv_adx = float(hadx.iloc[-1])
                h_rsi = None if np.isnan(hv_rsi) else hv_rsi
                h_adx = None if np.isnan(hv_adx) else hv_adx
            except Exception:
                h_close = h_sma20 = h_rsi = h_adx = None

        kapi = gates_6(direction, last_close, last_sma20, last_rsi, last_low, last_up, last_adx, h_close, h_sma20, h_rsi, h_adx)
        strong = is_strong(direction, int(skor), int(kapi))

        qv = compute_qv_24h(tickers.get(sym))

        base = sym.split("/")[0].strip().upper()
        rows.append(
            {
                "YÖN": direction,
                "COIN": base,
                "SKOR": int(skor),
                "FİYAT": float(last_close),
                "RAW": int(round(raw)),
                "QV_24H": float(qv),
                "KAPI": int(kapi),
                "STRONG": bool(strong),
                "EX": exchange_name.upper(),
            }
        )
        time.sleep(0.01)

    return pd.DataFrame(rows), meta


def merge_sources(df_k: pd.DataFrame, df_o: pd.DataFrame) -> pd.DataFrame:
    if df_k is None:
        df_k = pd.DataFrame()
    if df_o is None:
        df_o = pd.DataFrame()

    if df_k.empty and df_o.empty:
        return pd.DataFrame()

    # mark SOURCE= BOTH if coin appears on both
    coins_k = set(df_k["COIN"].tolist()) if not df_k.empty else set()
    coins_o = set(df_o["COIN"].tolist()) if not df_o.empty else set()
    both = coins_k.intersection(coins_o)

    def add_source(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out["SOURCE"] = np.where(out["COIN"].isin(both), "BOTH", out["EX"])
        return out.drop(columns=["EX"], errors="ignore")

    all_df = pd.concat([add_source(df_k), add_source(df_o)], ignore_index=True)

    # keep best row per coin (quality)
    src_rank = {"BOTH": 0, "KUCOIN": 1, "OKX": 2}
    tmp = all_df.copy()
    tmp["__src_rank"] = tmp["SOURCE"].map(src_rank).fillna(9).astype(int)
    tmp["__strong_rank"] = tmp["STRONG"].astype(bool).astype(int)
    tmp["__score_rank"] = np.where(tmp["YÖN"].eq("SHORT"), tmp["SKOR"].astype(int), -tmp["SKOR"].astype(int))

    tmp = tmp.sort_values(
        ["__strong_rank", "__src_rank", "KAPI", "__score_rank", "QV_24H"],
        ascending=[False, True, False, True, False],
        kind="mergesort",
    )

    tmp = tmp.drop_duplicates(subset=["COIN"], keep="first").drop(columns=["__src_rank", "__strong_rank", "__score_rank"])
    return tmp.reset_index(drop=True)


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    strong = df[df["STRONG"].astype(bool)].copy()
    non = df[~df["STRONG"].astype(bool)].copy()

    if not strong.empty:
        # strong first, but keep size TABLE_ROWS
        src_rank = {"BOTH": 0, "KUCOIN": 1, "OKX": 2}
        strong["__src_rank"] = strong["SOURCE"].map(src_rank).fillna(9).astype(int)
        strong = strong.sort_values(["__src_rank", "KAPI", "QV_24H"], ascending=[True, False, False]).drop(columns=["__src_rank"])
        out = strong.head(TABLE_ROWS)

        if len(out) < TABLE_ROWS:
            need = TABLE_ROWS - len(out)
            non["__src_rank"] = non["SOURCE"].map(src_rank).fillna(9).astype(int)
            non["__score_rank"] = np.where(non["YÖN"].eq("SHORT"), non["SKOR"].astype(int), -non["SKOR"].astype(int))
            non = non.sort_values(["__src_rank", "KAPI", "__score_rank", "QV_24H"], ascending=[True, False, True, False]).drop(columns=["__src_rank", "__score_rank"])
            out = pd.concat([out, non.head(need)], ignore_index=True)

        return out.reset_index(drop=True)

    # No STRONG: fallback 10+10
    longs = df[df["YÖN"].eq("LONG")].sort_values(["SKOR", "KAPI", "QV_24H"], ascending=[False, False, False]).head(FALLBACK_LONG)
    shorts = df[df["YÖN"].eq("SHORT")].sort_values(["SKOR", "KAPI", "QV_24H"], ascending=[True, False, False]).head(FALLBACK_SHORT)
    out = pd.concat([longs, shorts], ignore_index=True)

    # BOTH top
    src_rank = {"BOTH": 0, "KUCOIN": 1, "OKX": 2}
    out["__src_rank"] = out["SOURCE"].map(src_rank).fillna(9).astype(int)
    out["__score_rank"] = np.where(out["YÖN"].eq("SHORT"), out["SKOR"].astype(int), -out["SKOR"].astype(int))
    out = out.sort_values(["__src_rank", "KAPI", "__score_rank", "QV_24H"], ascending=[True, False, True, False]).drop(columns=["__src_rank", "__score_rank"])
    return out.head(TABLE_ROWS).reset_index(drop=True)


# -----------------------------
# Styling (dark) — keep base look
# -----------------------------
def style_table(df: pd.DataFrame):
    def row_style(row: pd.Series):
        yon = str(row.get("YÖN", ""))
        strong = bool(row.get("STRONG", False))
        source = str(row.get("SOURCE", ""))

        # LONG green / SHORT red
        if yon == "LONG":
            base = "rgba(6,78,59,0.35)"
            strong_bg = "#064e3b"
            both_bg = "#043227"
        else:
            base = "rgba(127,29,29,0.35)"
            strong_bg = "#7f1d1d"
            both_bg = "#4a0f0f"

        bg = strong_bg if strong else base
        if source == "BOTH":
            bg = both_bg

        return [f"background-color: {bg}; color: #e6edf3; font-weight: 700;"] * len(row)

    fmt = {
        "FİYAT": "{:.6f}",
        "QV_24H": "{:,.0f}",
        "RAW": "{:.0f}",
        "SKOR": "{:.0f}",
        "KAPI": "{:.0f}",
    }
    return df.style.format(fmt).apply(row_style, axis=1).set_properties(**{"border-color": "#1f2a37"})


# -----------------------------
# Telegram formatting
# -----------------------------
def format_report(df_show: pd.DataFrame) -> str:
    lines = []
    lines.append(f"📊 Gözcü Raporu (KuCoin+OKX) — {now_ist().strftime('%Y-%m-%d %H:%M:%S')} (IST)")
    lines.append(f"TF={TF}, HTF={HTF} | STRONG: KAPI=6 & (>=90 / <=10)")
    lines.append("")
    for _, r in df_show.iterrows():
        yon = r["YÖN"]
        coin = r["COIN"]
        skor = int(r["SKOR"])
        kapi = int(r["KAPI"])
        src = r["SOURCE"]
        strong = "🔥" if bool(r["STRONG"]) and yon == "LONG" else ("💀" if bool(r["STRONG"]) and yon == "SHORT" else "⏳")
        lines.append(f"{strong} {yon:<5} {coin:<10} SKOR={skor:<3} KAPI={kapi} SRC={src}")
    return "\n".join(lines)[:3900]


def format_strong_alert(row: pd.Series) -> tuple[str, str]:
    yon = row["YÖN"]
    coin = row["COIN"]
    skor = int(row["SKOR"])
    kapi = int(row["KAPI"])
    src = row["SOURCE"]
    tag = "🔥 STRONG LONG" if yon == "LONG" else "💀 STRONG SHORT"
    key = f"{coin}|{tag}"
    text = (
        f"{tag}\n"
        f"COIN: {coin}\n"
        f"SRC: {src}\n"
        f"SKOR: {skor}\n"
        f"KAPI: {kapi}\n"
        f"TF: {TF}  HTF: {HTF}\n"
        f"Time(IST): {now_ist().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return key, text


def now_ist() -> datetime:
    return datetime.now(IST)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎯 KuCoin PRO Sniper — Auto (LONG + SHORT) + Telegram")
st.caption(f"TF={TF} • HTF={HTF} • Auto refresh={AUTO_REFRESH_SEC}s • Report={REPORT_EVERY_MIN}m • Cooldown={ALERT_COOLDOWN_MIN}m")
st.markdown(f"**İstanbul Time:** {now_ist().strftime('%Y-%m-%d %H:%M:%S')}")

if "tg_last_report" not in st.session_state:
    st.session_state["tg_last_report"] = None
if "tg_last_alert_map" not in st.session_state:
    st.session_state["tg_last_alert_map"] = {}

# Auto refresh
try:
    st.autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
except Exception:
    try:
        st.experimental_autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh")
    except Exception:
        pass

# Scan
progress = st.progress(0, text="Başlıyor…")
status = st.empty()

with st.spinner("KuCoin + OKX taranıyor…"):
    t0 = time.time()

    # KuCoin
    df_k = pd.DataFrame()
    meta_k = {"universe": 0, "after_liq": 0, "scanned": 0}
    try:
        status.info("KuCoin taranıyor…")
        df_k, meta_k = scan_exchange("kucoin")
    except Exception:
        pass
    progress.progress(50, text="KuCoin tamam…")

    # OKX
    df_o = pd.DataFrame()
    meta_o = {"universe": 0, "after_liq": 0, "scanned": 0}
    try:
        status.info("OKX taranıyor…")
        df_o, meta_o = scan_exchange("okx")
    except Exception:
        pass
    progress.progress(85, text="OKX tamam…")

    status.info("Birleştiriliyor…")
    df_all = merge_sources(df_k, df_o)
    df_show = build_table(df_all)

    progress.progress(100, text="Bitti ✅")
    status.empty()

elapsed = int(max(1, time.time() - t0))

# Counters
if df_show is None or df_show.empty:
    st.warning("Aday yok. Bir sonraki yenilemeyi bekle.")
    st.stop()

strong_long = int(((df_show["YÖN"] == "LONG") & (df_show["STRONG"] == True)).sum())
strong_short = int(((df_show["YÖN"] == "SHORT") & (df_show["STRONG"] == True)).sum())
longs = int((df_show["YÖN"] == "LONG").sum())
shorts = int((df_show["YÖN"] == "SHORT").sum())

st.info(f"✅ STRONG LONG: {strong_long} | 💀 STRONG SHORT: {strong_short} | LONG: {longs} | SHORT: {shorts}")

evren_total = int(meta_k.get("universe", 0)) + int(meta_o.get("universe", 0))
after_liq = int(meta_k.get("after_liq", 0)) + int(meta_o.get("after_liq", 0))
scanned = int(meta_k.get("scanned", 0)) + int(meta_o.get("scanned", 0))
st.caption(f"Evren: {evren_total} • Likidite sonrası: {after_liq} • Tarama: {scanned} • Süre: {elapsed}s")

# Telegram send logic
if telegram_enabled():
    # Instant strong alerts (cooldown protected)
    strong_rows = df_show[df_show["STRONG"] == True].copy()
    if not strong_rows.empty:
        for _, row in strong_rows.iterrows():
            key, msg = format_strong_alert(row)
            if can_send_alert(key):
                ok, _ = telegram_send(TELEGRAM_BOT_TOKEN.strip(), TELEGRAM_CHAT_ID.strip(), msg)
                if ok:
                    mark_alert_sent(key)

    # 20 min report
    if can_send_report():
        rep = format_report(df_show)
        ok, _ = telegram_send(TELEGRAM_BOT_TOKEN.strip(), TELEGRAM_CHAT_ID.strip(), rep)
        if ok:
            mark_report_sent()
else:
    st.warning("Telegram kapalı: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID doldur (dosyanın en üstünde).")

# Show table
st.subheader("📋 SNIPER TABLO (Top 20)")
cols = ["YÖN", "COIN", "SKOR", "FİYAT", "RAW", "QV_24H", "KAPI", "STRONG", "SOURCE"]
df_out = df_show.loc[:, cols].copy()
st.dataframe(style_table(df_out), use_container_width=True, height=680)
