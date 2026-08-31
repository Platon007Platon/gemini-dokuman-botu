import os
import glob
import sqlite3
import numpy as np
import streamlit as st
from google import genai
from pypdf import PdfReader

# TF-IDF ve Benzerlik Araması İçin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SAYFA VE MENÜ AYARLARI ---
st.set_page_config(
    page_title="Belge Asistanı ✈️", 
    page_icon="✈️",
    layout="centered"
)

# --- 2. VERİTABANI (SQLITE) İŞLEMLERİ ---
DB_FILE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_message(role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def load_messages():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def clear_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages")
    conn.commit()
    conn.close()

init_db()

# --- CSS: YÖNETİCİ KUTUSU ---
st.markdown("""
    <style>
    .stAppDeployButton {display:none !important;}
    footer {visibility: hidden !important;}
    
    div[data-testid="stExpander"] {
        position: absolute !important;
        top: 10px !important;
        right: 60px !important;
        width: 240px !important;
        z-index: 999999 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
    }

    div[data-testid="stExpander"] summary {
        padding: 4px 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
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
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)

if not GEMINI_API_KEY:
    st.error("⚠️ API Anahtarı Streamlit Secrets üzerinde bulunamadı!")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# --- PDF İŞLEME VE METİN PARÇALAMA ---
mevcut_pdfler = glob.glob("*.pdf")

if "secili_pdfler" not in st.session_state:
    st.session_state.secili_pdfler = mevcut_pdfler.copy()

# --- YÖNETİCİ PANELİ ---
ADMIN_SIFRE = "13579S"

with st.expander("🔒 Yönetici / PDF Seçimi"):
    girilen_sifre = st.text_input("Şifre", type="password")
    
    if girilen_sifre == ADMIN_SIFRE:
        st.success("Giriş başarılı!")
        
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
        
        secilen_liste = []
        for pdf in mevcut_pdfler:
            varsayilan = pdf in st.session_state.secili_pdfler
            if st.checkbox(pdf, value=varsayilan, key=f"chk_{pdf}"):
                secilen_liste.append(pdf)
                
        if secilen_liste != st.session_state.secili_pdfler:
            st.session_state.secili_pdfler = secilen_liste
            st.cache_data.clear()
            st.rerun()

        st.divider()
        if st.button("🗑️ Sohbet Geçmişini Sıfırla"):
            clear_db()
            st.session_state.messages = []
            st.success("Geçmiş temizlendi!")
            st.rerun()

    elif girilen_sifre != "":
        st.error("Hatalı şifre!")

# --- PDF PARÇALAMA & TF-IDF ARAMA FONKSİYONLARI ---
@st.cache_data
def pdfleri_parcala(pdf_listesi, chunk_size=600, overlap=100):
    """PDF'leri okur ve küçük metin bloklarına böler."""
    if not pdf_listesi:
        return []
        
    metin_parcalari = []
    
    for pdf_yolu in pdf_listesi:
        try:
            reader = PdfReader(pdf_yolu)
            tam_dokuman_metni = ""
            for page in reader.pages:
                tam_dokuman_metni += (page.extract_text() or "") + " "
            
            start = 0
            while start < len(tam_dokuman_metni):
                end = start + chunk_size
                chunk = tam_dokuman_metni[start:end].strip()
                if len(chunk) > 30:
                    metin_parcalari.append(chunk)
                start += chunk_size - overlap
        except Exception:
            continue
            
    return metin_parcalari

def en_alakali_parcalari_bul(soru, metin_parcalari, top_k=5):
    """TF-IDF ile soruya en yakın metin parçalarını bulur."""
    if not metin_parcalari:
        return ""
        
    tum_icerik = [soru] + metin_parcalari
    vectorizer = TfidfVectorizer().fit_transform(tum_icerik)
    vectors = vectorizer.toarray()
    
    soru_vektoru = vectors[0].reshape(1, -1)
    parca_vektorleri = vectors[1:]
    
    benzerlikler = cosine_similarity(soru_vektoru, parca_vektorleri)[0]
    en_iyi_indeksler = np.argsort(benzerlikler)[::-1][:top_k]
    
    secilen_parcalar = []
    for idx in en_iyi_indeksler:
        if benzerlikler[idx] > 0.01:
            secilen_parcalar.append(metin_parcalari[idx])
            
    return "\n---\n".join(secilen_parcalar) if secilen_parcalar else metin_parcalari[0]

# --- BAŞLIK VE PDF YÜKLEME ---
st.markdown('<div class="big-title">Belge Asistanı ✈️</div>', unsafe_allow_html=True)

aktif_pdfler = st.session_state.get("secili_pdfler", [])
metin_parcalari = pdfleri_parcala(aktif_pdfler)

if not aktif_pdfler:
    st.warning("⚠️ Lütfen Yönetici panelinden soruların aranacağı en az 1 adet PDF seçin.")
    st.stop()

if not metin_parcalari:
    st.warning("Henüz sistemde okunabilir bir döküman bulunmuyor.")
    st.stop()

# --- SOHBET GEÇMİŞİ YÜKLEME ---
if "messages" not in st.session_state:
    st.session_state.messages = load_messages()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.caption("Made by Serd@R T.")

# --- SOHBET İŞLEMLERİ ---
if kullanici_sorusu := st.chat_input("Ne öğrenmek istiyorsun?"):
    
    st.chat_message("user").markdown(kullanici_sorusu)
    st.session_state.messages.append({"role": "user", "content": kullanici_sorusu})
    save_message("user", kullanici_sorusu)

    sadece_alakali_baglam = en_alakali_parcalari_bul(kullanici_sorusu, metin_parcalari, top_k=5)

    prompt_metni = f"""
Sen "Belge Asistanı ✈️" adında zeki bir asistansın.

GİZLİLİK KURALI:
Sistemdeki dosya isimlerini (örneğin ACC.pdf vb.) kesinlikle açıklama. Dosya ismi sorulursa "Güvenlik nedeniyle dosya bilgilerini paylaşamıyorum." de.

TALİMATLAR:
1. Sana verilen İLGİLİ DÖKÜMAN PARÇALARI'nı dikkatlice incele.
2. Kullanıcının sorusu döküman parçalarında DOĞRUDAN VARSA, net şekilde cevapla.
3. Kullanıcının sorduğu soru verilen parçalarda DOĞRUDAN YOKSA:
   - İlk satıra TAM OLARAK şunu yaz: "Aradığınız bilgi dökümanda doğrudan bulunamadı."
   - Ardından döküman içeriklerinde yer alan ve kullanıcının sorusuyla en ilişkili konuyu ekle.

İLGİLİ DÖKÜMAN PARÇALARI:
{sadece_alakali_baglam}

KULLANICI SORUSU:
{kullanici_sorusu}
"""

    with st.chat_message("assistant"):
        with st.spinner("İlgili paragraflar taranıyor..."):
            try:
                # DÜZELTME: Model ismi güncellendi
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=prompt_metni
                )
                cevap = response.text
                st.markdown(cevap)
                
                st.session_state.messages.append({"role": "assistant", "content": cevap})
                save_message("assistant", cevap)
                
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")