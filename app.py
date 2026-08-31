import os
import glob
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
from pypdf import PdfReader

# --- 1. SAYFA VE MENÜ AYARLARI ---
st.set_page_config(
    page_title="Belge Asistanı  ✈️ ", 
    page_icon="✈️ ",
    layout="centered"
)

# --- CSS: YÖNETİCİ KUTUSUNU SAĞ ÜST KÖŞEYE SABİTLEME ---
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
        width: 220px !important;
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

# --- VEKTÖR EMBEDDING MODELİNİ YÜKLE ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

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

# --- KLASÖRDEKİ TÜM PDF'LERİ BULMA ---
mevcut_pdfler = glob.glob("*.pdf")

# Session state üzerinden seçili PDF'leri yönetme
if "secili_pdfler" not in st.session_state:
    st.session_state.secili_pdfler = mevcut_pdfler.copy()

# --- SAĞ ÜSTE SABİTLENMİŞ YÖNETİCİ PANELİ ---
ADMIN_SIFRE = "13579S"  # Yönetici Şifren

with st.expander("🔒 Yönetici / PDF Seçimi"):
    girilen_sifre = st.text_input("Şifre", type="password")
    
    if girilen_sifre == ADMIN_SIFRE:
        st.success("Giriş başarılı!")
        
        # 1. Yeni Dosya Yükleme Alanı
        yuklenen_dosya = st.file_uploader("Yeni PDF ekle", type=["pdf"])
        if yuklenen_dosya:
            dosya_adi = yuklenen_dosya.name
            with open(dosya_adi, "wb") as f:
                f.write(yuklenen_dosya.getbuffer())
            st.cache_data.clear()
            if dosya_adi not in st.session_state.secili_pdfler:
                st.session_state.secili_pdfler.append(dosya_adi)
            st.success(f"'{dosya_adi}' yüklendi!")
            st.rerun()

        st.divider()
        st.markdown("**Aktif Edilecek Belgeleri Seç:**")
        
        # 2. PDF Seçim Kutuları (Checkboxes)
        secilen_liste = []
        for pdf in mevcut_pdfler:
            varsayilan = pdf in st.session_state.secili_pdfler
            if st.checkbox(pdf, value=varsayilan, key=f"chk_{pdf}"):
                secilen_liste.append(pdf)
                
        # Seçimleri güncelle ve önbelleği sıfırla
        if secilen_liste != st.session_state.secili_pdfler:
            st.session_state.secili_pdfler = secilen_liste
            st.cache_data.clear()

    elif girilen_sifre != "":
        st.error("Hatalı şifre!")

# --- SEÇİLİ PDF'LERİ VEKTÖRLEŞTİRME VE İNDEKSLEME ---
@st.cache_data
def secili_pdfleri_vektorlestir(pdf_listesi, chunk_size=700, overlap=100):
    if not pdf_listesi:
        return None, None
        
    parcalar = []
    for pdf_yolu in pdf_listesi:
        try:
            reader = PdfReader(pdf_yolu)
            metin = ""
            for page in reader.pages:
                metin += page.extract_text() or ""
            
            # Metni küçük parçalara bölme
            for i in range(0, len(metin), chunk_size - overlap):
                parca = metin[i:i + chunk_size]
                if parca.strip():
                    parcalar.append(parca)
        except Exception:
            continue
            
    if not parcalar:
        return None, None

    # FAISS Vektör İndeksi Oluşturma
    embeddings = embedder.encode(parcalar, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    return index, parcalar

# --- BAŞLIK ---
st.markdown('<div class="big-title">Belge Asistanı  ✈️ </div>', unsafe_allow_html=True)

# Vektör Veritabanı Hazırlığı
aktif_pdfler = st.session_state.get("secili_pdfler", [])
index, parcalar = secili_pdfleri_vektorlestir(aktif_pdfler)

if not aktif_pdfler:
    st.warning("⚠️ Lütfen Yönetici panelinden soruların aranacağı en az 1 adet PDF seçin.")
    st.stop()

if index is None or parcalar is None:
    st.warning("Henüz sistemde okunabilir bir döküman bulunmuyor.")
    st.stop()

# --- SOHBET GEÇMİŞİ (CHAT HISTORY) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana yazdır
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.caption("Made by Serd@R T.")

# --- SOHBET İŞLEMLERİ (VEKTÖR RAG SORGULAMA) ---
if kullanici_sorusu := st.chat_input("Ne öğrenmek istiyorsun?"):
    
    # Kullanıcı mesajını ekrana ve hafızaya ekle
    st.chat_message("user").markdown(kullanici_sorusu)
    st.session_state.messages.append({"role": "user", "content": kullanici_sorusu})

    # Vektör Arama: Soru ile en alakalı 4 metin parçasını bul
    soru_vektoru = embedder.encode([kullanici_sorusu], convert_to_numpy=True)
    distances, indices = index.search(np.array(soru_vektoru).astype('float32'), k=min(4, len(parcalar)))
    
    alakali_baglam = "\n---\n".join([parcalar[i] for i in indices[0] if i < len(parcalar)])

    prompt_metni = f"""
Sen "Belge Asistanı  ✈️ " adında zeki bir asistansın.

GİZLİLİK KURALI:
Sistemdeki dosya isimlerini (örneğin ACC.pdf vb.) kesinlikle açıklama. Dosya ismi sorulursa "Güvenlik nedeniyle dosya bilgilerini paylaşamıyorum." de.

TALİMATLAR:
1. Sana verilen DÖKÜMAN İÇERİKLERİ'ni (birden fazla döküman içerebilir) bütünsel olarak incele.
2. Kullanıcının sorusu dökümanların herhangi birinde DOĞRUDAN VARSA, net şekilde cevapla.
3. Kullanıcının sorduğu soru dökümanların hiçbirinde DOĞRUDAN YOKSA:
   - İlk satıra TAM OLARAK şunu yaz: "Aradığınız bilgi dökümanda doğrudan bulunamadı."
   - Ardından döküman içeriklerinde yer alan ve kullanıcının sorusuyla en çok ilişkili/yakın olan konuyu bul.
   - Şu formatta yanıtla:

Aradığınız bilgi dökümanda doğrudan bulunamadı.

---
💡 **İlişkili Olabilecek Konu ve Yanıt:**
*(İlişkili konu başlığı)*
*(O konuyla ilgili dökümandaki açıklama)*

DÖKÜMAN İÇERİKLERİ:
{alakali_baglam}

KULLANICI SORUSU:
{kullanici_sorusu}
"""

    with st.chat_message("assistant"):
        with st.spinner("Seçili belgeler taranıyor..."):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt_metni
                )
                cevap = response.text
                st.markdown(cevap)
                st.session_state.messages.append({"role": "assistant", "content": cevap})
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")