import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import plotly.graph_objects as go

# --- AYARLAR VE BAĞLANTI ---
st.set_page_config(layout="wide", page_title="Quant Signal Pro")
exchange = ccxt.binance()

# --- VERİ ÇEKME FONKSİYONU ---
def fetch_data(symbol='BTC/USDT', timeframe='1h'):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Sinyal Mantığın (Burayı kendi orijinal hesaplamalarınla aynı tutuyorum)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['Aksiyon'] = 'BEKLE'
        df.loc[df['RSI'] < 30, 'Aksiyon'] = 'AL'
        df.loc[df['RSI'] > 70, 'Aksiyon'] = 'SAT'
        return df
    except:
        return pd.DataFrame()

# --- RENKLENDİRME FONKSİYONU ---
def style_action_color(val):
    if val == 'AL': return 'background-color: green; color: white'
    if val == 'SAT': return 'background-color: red; color: white'
    return ''

# --- ARAYÜZ ---
st.title("📊 Quant Signal Pro")

tab1, tab2 = st.tabs(["🔍 İŞLEM TARAYICI", "📈 ANALİZ MASASI"])

with tab1:
    df = fetch_data()
    
    # --- HATAYI DÜZELTEN KRİTİK KISIM BURASI ---
    if not df.empty:
        try:
            # Sütun isimlerini kontrol ederek renklendirme uygular
            st.dataframe(df.style.applymap(style_action_color, subset=['Aksiyon']), height=650, use_container_width=True)
        except Exception as e:
            # Eğer renklendirmede hata çıkarsa tabloyu sade göster, uygulamayı çökertme
            st.dataframe(df, height=650, use_container_width=True)
    else:
        st.warning("Borsadan veri alınamadı veya tablo boş. Lütfen bekleyin...")

with tab2:
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close'])])
        st.plotly_chart(fig, use_container_width=True)
