import os
import streamlit as st
from google import genai

# Sayfa Ayarları
st.set_page_config(page_title="Doküman Asistanı", page_icon="🤖")
st.title("🤖 Gemini Doküman Asistanı")

# API Anahtarı Önceliği:
# 1. Streamlit Secrets (Bulut için)
# 2. Doğrudan API Key (Kendi bilgisayarında test için)
YEDEK_API_KEY = ""

try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", YEDEK_API_KEY)
except Exception:
    GEMINI_API_KEY = YEDEK_API_KEY

if not GEMINI_API_KEY:
    st.error("⚠️ API Anahtarı bulunamadı!")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# bilgi.txt okuma
@st.cache_data
def dokuman_oku():
    try:
        with open("dokuman.txt", "r", encoding="utf-8") as dosya:
            return dosya.read()
    except FileNotFoundError:
        return None

dosya_icerigi = dokuman_oku()

if dosya_icerigi is None:
    st.error("❌ 'bilgi.txt' dosyası bulunamadı! Lütfen aynı klasöre ekleyin.")
else:
    st.success("📄 Doküman başarıyla yüklendi!")
    
    kullanici_sorusu = st.text_input("Doküman hakkında ne öğrenmek istiyorsun?", placeholder="Örn: Ahmet kaç doğumludur?")

    if st.button("Soruyu Gönder", type="primary"):
        if kullanici_sorusu.strip() == "":
            st.warning("Lütfen bir soru yazın.")
        else:
            with st.spinner("Gemini dokümanı inceliyor..."):
                prompt_metni = (
                    f"Sen bir doküman asistanısın. SADECE verilen metne göre cevap ver. "
                    f"Metinde yoksa 'Bu bilgi dokümanda yok' de.\n\n"
                    f"DOKÜMAN İÇERİĞİ:\n{dosya_icerigi}\n\n"
                    f"SORU: {kullanici_sorusu}"
                )
                
                try:
                    response = client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=prompt_metni
                    )
                    
                    st.markdown("### 💡 Cevap:")
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Bir hata oluştu: {e}")