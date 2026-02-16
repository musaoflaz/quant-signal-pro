import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import plotly.graph_objects as go

# 1. Konfigürasyon
st.set_page_config(layout="wide", page_title="Quant Signal Pro V2")

# 2. Borsa Bağlantısı
exchange = ccxt.binance({'enableRateLimit': True})

def veri_getir(sembol='BTC/USDT'):
    try:
        bars = exchange.fetch_ohlcv(sembol, timeframe='1h', limit=100)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['Tarih'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Sinyal Hesaplama
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # Sütun ismi hatasını (KeyError) önlemek için sabit isim:
        df['SINYAL'] = 'BEKLE'
        df.loc[df['RSI'] < 30, 'SINYAL'] = 'AL'
        df.loc[df['RSI'] > 70, 'SINYAL'] = 'SAT'
        
        return df[['Tarih', 'open', 'high', 'low', 'close', 'RSI', 'SINYAL']].dropna()
    except:
        return pd.DataFrame()

def sinyal_stili(val):
    if val == 'AL': return 'background-color: #00ff00; color: black; font-weight: bold'
    if val == 'SAT': return 'background-color: #ff0000; color: white; font-weight: bold'
    return ''

# --- ARAYÜZ ---
st.title("📊 Quant Signal Pro")

df = veri_getir()

tab1, tab2 = st.tabs(["🔍 Sinyal Tarayıcı", "📈 Analiz Grafiği"])

with tab1:
    if not df.empty:
        # En güncel veriyi en üste alarak göster
        st.dataframe(
            df.iloc[::-1].style.applymap(sinyal_stili, subset=['SINYAL']),
            use_container_width=True,
            height=600
        )
    else:
        st.warning("Veriler yükleniyor, lütfen bekleyin...")

with tab2:
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df['Tarih'], open=df['open'], high=df['high'], low=df['low'], close=df['close']
        )])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

if st.sidebar.button('Yeniden Tara'):
    st.rerun()
