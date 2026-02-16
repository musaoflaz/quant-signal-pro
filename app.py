import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime
import time

# ==============================================================================
# 1. KURUMSAL TERMİNAL YAPILANDIRMASI
# ==============================================================================
st.set_page_config(
    page_title="QuantSignal Pro | İşlem Terminali",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #d1d4dc; }
    .stMetric { background-color: #1a1e26; padding: 10px; border-radius: 4px; border-left: 3px solid #3b82f6; }
    .live-badge { 
        padding: 4px 12px; border-radius: 20px; background-color: #16a34a; 
        color: white; font-size: 0.75rem; font-weight: bold; animation: pulse 2s infinite;
        display: inline-block; box-shadow: 0 0 10px rgba(22, 163, 74, 0.5);
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
    .status-card {
        padding: 12px; border-radius: 8px; background: #1a1e26; border: 1px solid #30363d;
        margin-bottom: 10px; font-size: 0.85rem;
    }
    .regime-trend { color: #3b82f6; font-weight: bold; }
    .regime-range { color: #f59e0b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. GELİŞMİŞ QUANT MOTORU (TRADE ENGINE)
# ==============================================================================
class LiveQuantEngine:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True, 
            'options': {'defaultType': 'future'}
        })

    @st.cache_data(ttl=300)
    def get_market_universe(_self):
        try:
            tickers = _self.exchange.fetch_tickers()
            markets = _self.exchange.load_markets()
            data = []
            for s, m in markets.items():
                if m['linear'] and m['quote'] == 'USDT' and m['active']:
                    tick = tickers.get(s, {})
                    data.append({
                        'symbol': s, 'price': tick.get('last', 0),
                        'change': tick.get('percentage', 0), 'volume': tick.get('quoteVolume', 0)
                    })
            return pd.DataFrame(data).sort_values('volume', ascending=False)
        except: return pd.DataFrame()

    def fetch_live_data(self, symbol, timeframe='15m', limit=400):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            return df
        except: return pd.DataFrame()

    def apply_strategy(self, df):
        if df.empty or len(df) < 200: return df
        try:
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.ema(length=200, append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.adx(length=14, append=True)
            df.ta.macd(append=True)
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma']
        except: pass
        return df

    def get_signal_logic(self, d15, d4h):
        if d15.empty or d4h.empty or 'EMA_200' not in d4h.columns:
            return "VERİ BEKLENİYOR", 0, "Analiz hazırlanıyor...", "N/A"
        
        l15 = d15.iloc[-1]
        l4 = d4h.iloc[-1]
        
        # PİYASA REJİMİ TESPİTİ
        adx = l15.get('ADX_14', 0)
        regime = "TREND (TRENDİNG)" if adx > 25 else "YATAY (RANGİNG)"
        
        score = 0
        reasons = []
        
        # Trend Onayı
        if float(l4['close']) > float(l4['EMA_200']):
            score += 30; reasons.append("H4 Boğa")
        else:
            score -= 30; reasons.append("H4 Ayı")

        # RSI & MACD
        rsi = l15.get('RSI_14', 50)
        if rsi > 60: score += 20; reasons.append("RSI+")
        elif rsi < 40: score -= 20; reasons.append("RSI-")

        macd_h = l15.get('MACDh_12_26_9', 0)
        if macd_h > 0: score += 15; reasons.append("MACD+")
        else: score -= 15; reasons.append("MACD-")

        # KARAR MEKANİZMASI (NET İŞLEM EYLEMİ)
        final_score = np.clip(score, -100, 100)
        
        if final_score >= 50: sig = "🚀 GÜÇLÜ AL (LONG)"
        elif final_score >= 15: sig = "🟢 AL (LONG)"
        elif final_score <= -50: sig = "🔥 GÜÇLÜ SAT (SHORT)"
        elif final_score <= -15: sig = "🔴 SAT (SHORT)"
        else: sig = "⚪ BEKLE (NÖTR)"
        
        return sig, abs(round(final_score)), " | ".join(reasons), regime

# ==============================================================================
# 3. UI LİVE TERMİNAL
# ==============================================================================
def main():
    engine = LiveQuantEngine()
    
    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
    with col_h1:
        st.markdown("<h1 style='margin:0;'>🏛️ TRADE TERMINAL</h1>", unsafe_allow_html=True)
    with col_h2:
        auto_refresh = st.toggle("Otomatik Canlı Akış", value=True)
    with col_h3:
        st.markdown(f"<div style='text-align: right;'><span class='live-badge'>ONLINE</span><br><small>{datetime.now().strftime('%H:%M:%S')}</small></div>", unsafe_allow_html=True)

    tab_scan, tab_chart = st.tabs(["🔍 İŞLEM TARAYICI", "📊 ANALİZ MASASI"])

    with tab_scan:
        uni = engine.get_market_universe().head(20)
        results = []
        prog = st.progress(0)
        
        for i, (idx, row) in enumerate(uni.iterrows()):
            sym = row['symbol']
            d15 = engine.apply_strategy(engine.fetch_live_data(sym, '15m'))
            d4h = engine.apply_strategy(engine.fetch_live_data(sym, '4h'))
            sig, score, reason, regime = engine.get_signal_logic(d15, d4h)
            
            results.append({
                'VARLIK': sym,
                'FİYAT': f"${row['price']:.4f}",
                'PİYASA REJİMİ': regime,
                'İŞLEM EYLEMİ': sig,
                'GÜVEN %': f"{score}%",
                'TEKNİK ANALİZ': reason
            })
            prog.progress((i + 1) / len(uni))
        
        df = pd.DataFrame(results)
        
        def style_action_col(val):
            if "AL (LONG)" in val: return 'background-color: #064e3b; color: #10b981; font-weight: bold;'
            if "SAT (SHORT)" in val: return 'background-color: #450a0a; color: #ef4444; font-weight: bold;'
            return 'color: #848e9c;'

        st.dataframe(
            df.style.applymap(style_action_col, subset=['İŞLEM EYLEMİ']),
            use_container_width=True,
            hide_index=True,
            height=650
        )

    with tab_chart:
        c1, c2 = st.columns([3, 1])
        selected = c2.selectbox("Varlık Seçin", uni['symbol'].tolist() if not uni.empty else ["BTC/USDT"])
        
        d15_s = engine.apply_strategy(engine.fetch_live_data(selected, '15m'))
        d4h_s = engine.apply_strategy(engine.fetch_live_data(selected, '4h'))
        sig, score, reason, regime = engine.get_signal_logic(d15_s, d4h_s)
        
        with c2:
            st.markdown(f"**Piyasa Rejimi:** <br><span class='regime-{'trend' if 'TREND' in regime else 'range'}'>{regime}</span>", unsafe_allow_html=True)
            
            fig_g = go.Figure(go.Indicator(
                mode = "gauge+number", value = score,
                title = {'text': sig, 'font': {'size': 18}},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#3b82f6"}}
            ))
            fig_g.update_layout(height=250, margin=dict(t=30, b=0, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", font={'color':"white"})
            st.plotly_chart(fig_g, use_container_width=True)
            
            st.markdown(f"<div class='status-card'><b>Teknik Gerekçeler:</b><br>{reason}</div>", unsafe_allow_html=True)

        with c1:
            if not d15_s.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=d15_s['ts'], open=d15_s['open'], high=d15_s['high'], low=d15_s['low'], close=d15_s['close'], name="Mumlar"))
                fig.add_trace(go.Scatter(x=d15_s['ts'], y=d15_s['EMA_200'], name="Ana Trend (200)", line=dict(color='#f59e0b', width=2)))
                
                fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=600, paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14")
                st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(15)
        st.rerun()

if __name__ == "__main__":
    main()
