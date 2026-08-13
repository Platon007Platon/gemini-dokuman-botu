import os
import streamlit as st
from google import genai
from pypdf import PdfReader

# --- 1. SAYFA VE MENÜ AYARLARI ---
st.set_page_config(
    page_title="Belge Asistanı  ✈️ ", 
    page_icon="✈️ ",
    layout="centered"
)

# --- CSS: YÖNETİCİ KUTUSUNU SAĞ ÜST KÖŞEYE (ÜÇ NOKTANIN ALTINA/YANINA) SABİTLEME ---
st.markdown("""
    <style>
    /* GitHub Deploy butonunu ve varsayılan footer'ı gizle */
    .stAppDeployButton {display:none !important;}
    footer {visibility: hidden !important;}
    
    /* Yönetici kutusunu ekranın sağ üst köşesine sabitleme */
    div[data-testid="stExpander"] {
        position: absolute !important;
        top: 10px !important;
        right: 60px !important;
        width: 170px !important;
        z-index: 999999 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
    }

    /* Expander içi yazıların düzeni */
    div[data-testid="stExpander"] summary {
        padding: 4px 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    /* Başlık stili */
    .big-title {
        font-size: 2.2rem !important;
        font-weight: 800;
        color: #0F172A;
        margin-top: 40px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- GÜVENLİK KONTROLÜ (API KEY) ---
YEDEK_API_KEY = ""
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", YEDEK_API_KEY)
except Exception:
    GEMINI_API_KEY = YEDEK_API_KEY

if not GEMINI_API_KEY:
    st.error("⚠️ API Anahtarı bulunamadı!")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# --- SAĞ ÜSTE SABİTLENMİŞ YÖNETİCİ PANELİ ---
ADMIN_SIFRE = "13579S"  # Yönetici Şifren

with st.expander("🔒 Yönetici"):
    girilen_sifre = st.text_input("Şifre", type="password")
    
    if girilen_sifre == ADMIN_SIFRE:
        st.success("Giriş başarılı!")
        yuklenen_dosya = st.file_uploader("Yeni bir PDF yükleyin", type=["pdf"])
        if yuklenen_dosya:
            with open("ACC.pdf", "wb") as f:
                f.write(yuklenen_dosya.getbuffer())
            st.success("Yeni dosya kaydedildi! Sayfayı yenileyin.")
    elif girilen_sifre != "":
        st.error("Hatalı şifre!")

# --- BAŞLIK ---
st.markdown('<div class="big-title">Belge Asistanı  ✈️ </div>', unsafe_allow_html=True)

# --- PDF OKUMA FONKSİYONU ---
@st.cache_data
def pdf_oku(pdf_yolu):
    try:
        reader = PdfReader(pdf_yolu)
        metin = ""
        for page in reader.pages:
            metin += page.extract_text() or ""
        return metin
    except Exception:
        return None

dosya_icerigi = pdf_oku("ACC.pdf")

if dosya_icerigi is None:
    st.warning("Henüz sistemde yüklü bir döküman bulunmuyor.")
    st.stop()

# --- SOHBET GEÇMİŞİ (CHAT HISTORY) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana yazdır
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.caption("Made by Serd@R T.")

# --- SOHBET İŞLEMLERİ ---
if kullanici_sorusu := st.chat_input("Ne öğrenmek istiyorsun?"):
    
    # Kullanıcı mesajını ekrana ve hafızaya ekle
    st.chat_message("user").markdown(kullanici_sorusu)
    st.session_state.messages.append({"role": "user", "content": kullanici_sorusu})

    prompt_metni = f"""
Sen "Belge Asistanı  ✈️ " adında zeki bir asistansın.

GİZLİLİK KURALI:
Sistemdeki dosya isimlerini (örneğin ACC.pdf vb.) kesinlikle açıklama. Dosya ismi sorulursa "Güvenlik nedeniyle dosya bilgilerini paylaşamıyorum." de.

TALİMATLAR:
1. Sana verilen DÖKÜMAN İÇERİĞİ'ni incele.
2. Kullanıcının sorusu dökümanda DOĞRUDAN VARSA, net şekilde cevapla.
3. Kullanıcının sorduğu soru dökümanda DOĞRUDAN YOKSA:
   - İlk satıra TAM OLARAK şunu yaz: "Aradığınız bilgi dökümanda doğrudan bulunamadı."
   - Ardından döküman içeriğinde yer alan ve kullanıcının sorusuyla en çok ilişkili/yakın olan konuyu bul.
   - Şu formatta yanıtla:

Aradığınız bilgi dökümanda doğrudan bulunamadı.

---
💡 **İlişkili Olabilecek Konu ve Yanıt:**
*(İlişkili konu başlığı)*
*(O konuyla ilgili dökümandaki açıklama)*

DÖKÜMAN İÇERİĞİ:
{dosya_icerigi}

KULLANICI SORUSU:
{kullanici_sorusu}
"""

    with st.chat_message("assistant"):
        with st.spinner("İnceleniyor..."):
            try:
                response = client.models.generate_content(
                    model='gemini-3.5-flash',
                    contents=prompt_metni
                )
                cevap = response.text
                st.markdown(cevap)
                st.session_state.messages.append({"role": "assistant", "content": cevap})
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")