import streamlit as st
import pandas as pd
import ccxt
import pandas_ta as ta
import time
import requests

# --- SENİN BİLGİLERİN SİSTEME GÖMÜLDÜ ---
TOKEN = "8330775219:AAHMGpdCdCEStj-B4Y3_WHD7xPEbjeaHWFM"
CHAT_ID = "1358384022"

def telegram_yolla(mesaj):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": mesaj}, timeout=10)
    except Exception as e:
        st.error(f"Telegram Hatası: {e}")

# Borsa Bağlantısı (Kucoin/Binance Verisi)
exchange = ccxt.kucoin({'enableRateLimit': True})

st.set_page_config(page_title="Alpha Sniper V42", layout="wide")
st.title("🛡️ ALPHA SNIPER V42")
st.subheader("Otomatik Piyasa Gözcüsü")

# Bot Durum Yönetimi
if 'bot_calisiyor' not in st.session_state:
    st.session_state.bot_calisiyor = False

col1, col2 = st.columns(2)
with col1:
    if st.button("🟢 SİSTEMİ BAŞLAT"):
        st.session_state.bot_calisiyor = True
        telegram_yolla("🚀 Sniper Bot Aktif! 100 Puanlık 'Altın Sinyal' Bekleniyor...")
        st.success("Bağlantı Kuruldu! Telegram'ı kontrol et.")

with col2:
    if st.button("🔴 SİSTEMİ DURDUR"):
        st.session_state.bot_calisiyor = False
        st.warning("Sistem Durduruldu.")

# Ana Tarama Fonksiyonu
def tarama_baslat():
    # Binance 3x popüler coinler
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LTC/USDT', 'AVAX/USDT', 'FET/USDT', 'SUI/USDT', 'NEAR/USDT']
    
    for s in symbols:
        try:
            # Veri çekme
            bars = exchange.fetch_ohlcv(s, timeframe='1h', limit=150)
            df = pd.DataFrame(bars, columns=['t','o','h','l','c','v'])
            
            # Teknik Analiz (EMA + RSI + STOCH RSI)
            df['EMA200'] = ta.ema(df['c'], length=200) or df['c'].rolling(100).mean()
            df['RSI'] = ta.rsi(df['c'], length=14)
            stoch = ta.stochrsi(df['c'], length=14, rsi_length=14, k=3, d=3)
            df = pd.concat([df, stoch], axis=1)
            
            l = df.iloc[-1]  # Son mum
            p = df.iloc[-2]  # Önceki mum
            
            # Kolon isimlerini otomatik bul (Hata almamak için)
            sk = [c for c in df.columns if 'STOCHRSIk' in c][0]
            sd = [c for c in df.columns if 'STOCHRSId' in c][0]
            
            # 🎯 100 PUANLIK SÜPER SİNYAL STRATEJİSİ
            # 1. Şart: Fiyat EMA200 üzerinde (Yükselen Trend)
            # 2. Şart: Stoch RSI altta yukarı kesişim (Altın Kesişim)
            # 3. Şart: RSI aşırı şişmemiş (40-65 arası)
            
            skor = 0
            if l['c'] > l['EMA200']:
                skor += 40
                if p[sk] < p[sd] and l[sk] > l[sd]:
                    skor += 40
                if 40 <= l['RSI'] <= 65:
                    skor += 20
            
            if skor >= 100:
                mesaj = (f"🎯 **100 PUANLIK SİNYAL!**\n\n"
                         f"Coin: {s}\n"
                         f"Fiyat: {l['c']}\n"
                         f"RSI: {int(l['RSI'])}\n"
                         f"Durum: EMA Üstü + Stoch Kesişimi\n\n"
                         f"🚀 Binance 3x Hazır Ol!")
                telegram_yolla(mesaj)
                st.info(f"✅ Sinyal Gönderildi: {s}")
            
            time.sleep(0.1)
        except:
            continue

# Döngü
if st.session_state.bot_calisiyor:
    placeholder = st.empty()
    while st.session_state.bot_calisiyor:
        with placeholder.container():
            st.write(f"🔄 Tarama yapılıyor... Son Güncelleme: {time.strftime('%H:%M:%S')}")
            tarama_baslat()
            st.write("😴 5 dakika mola. Pusuya devam...")
            time.sleep(300)
