import os
import streamlit as st
from google import genai
from pypdf import PdfReader

# Sayfa Ayarları
st.set_page_config(page_title="Doküman Asistanı", page_icon="🤖")
st.title("🤖 Gemini Doküman Asistanı")

# API Anahtarı Önceliği (Streamlit Secrets / Yedek)
YEDEK_API_KEY = ""

try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", YEDEK_API_KEY)
except Exception:
    GEMINI_API_KEY = YEDEK_API_KEY

if not GEMINI_API_KEY:
    st.error("⚠️ API Anahtarı bulunamadı! Lütfen Streamlit Secrets ayarlarını kontrol edin.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# PDF okuma fonksiyonu
@st.cache_data
def pdf_oku(pdf_yolu):
    try:
        reader = PdfReader(pdf_yolu)
        metin = ""
        for page in reader.pages:
            metin += page.extract_text() + "\n"
        return metin
    except Exception as e:
        return None

# Okunacak dosya adı (ACC.pdf)
dosya_icerigi = pdf_oku("ACC.pdf")

if dosya_icerigi is None:
    st.error("❌ 'ACC.pdf' dosyası bulunamadı! Lütfen GitHub deposuna bu isimde dosyayı eklediğinizden emin olun.")
else:
    st.success("📄 'ACC.pdf' dokümanı başarıyla yüklendi!")
    
    kullanici_sorusu = st.text_input("Doküman hakkında ne öğrenmek istiyorsun?", placeholder="Örn: Dokümandaki ana konular nelerdir?")

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