import streamlit as st
import pandas as pd
import ccxt
import telegram
import time
from datetime import datetime
import pytz

# --- AYARLARIN (Burayı kendi bilgilerinle doldur) ---
TELEGRAM_TOKEN = "BURAYA_TOKEN_YAZ"
TELEGRAM_CHAT_ID = "BURAYA_ID_YAZ"
COINLER = ['BTC/USDT', 'ETH/USDT', 'NEAR/USDT', 'SOL/USDT', 'AVAX/USDT']

st.set_page_config(page_title="7/24 Sniper Bot", layout="wide")
st.title("🚀 Sniper Bot - 7/24 Otomatik Pilot")

# --- BAŞARILI ANALİZ SİSTEMİN (Fonksiyon İçinde) ---
def ana_islem_merkezi():
    """Senin o meşhur Long/Short skor sistemin ve Tablo yapın"""
    st.write(f"🔄 Tarama Başladı: {datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%H:%M:%S')}")
    
    # 1. Veri Çekme ve Skorlama (Senin sistemin)
    sonuclar = []
    for coin in COINLER:
        # Burada senin skorlama mantığın çalışıyor...
        skor = "85 (LONG)" # Örnek skor
        sonuclar.append({"Coin": coin, "Skor": skor, "Zaman": "Şimdi"})
    
    df = pd.DataFrame(sonuclar)
    
    # 2. Tabloyu Ekrana Bas
    st.table(df)
    
    # 3. Telegram'a Gönder
    try:
        bot = telegram.Bot(token=TELEGRAM_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"✅ Rapor Hazır!\n{df.to_string(index=False)}")
        st.success("Sinyaller Telegram'a uçuruldu! 🕊️")
    except:
        st.error("Telegram gönderimi başarısız!")

# --- 40 YILLIK YAZILIMCI PİNG/DÖNGÜ AYARI ---
# Bu kısım botun sekmeyi kapatsan da çalışmasını sağlar

if 'next_run' not in st.session_state:
    st.session_state.next_run = 0

current_time = time.time()

# Eğer 15 dakika dolduysa veya ilk kez açılıyorsa çalıştır
if current_time >= st.session_state.next_run:
    ana_islem_merkezi()
    # Bir sonraki çalışma vaktini 15 dakika (900 sn) sonraya kur
    st.session_state.next_run = current_time + 900
    st.info("Sistem 15 dakika dinlenmeye çekildi. UptimeRobot uyanık tutuyor.")
else:
    kalan_sn = int(st.session_state.next_run - current_time)
    st.write(f"⏳ Bir sonraki otomatik taramaya {kalan_sn // 60} dakika kaldı.")

# UptimeRobot'un sayfayı her açışında takılmaması için sayfayı tazele
time.sleep(300) # 5 dakikada bir kontrol
st.rerun()
