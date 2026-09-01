import os
import glob
import sqlite3
import re
import numpy as np
import streamlit as st
from google import genai
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SAYFA VE MENÜ AYARLARI ---
st.set_page_config(
    page_title="Belge Asistanı  ✈️ ",
    page_icon="✈️ ",
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

# --- CSS: YÖNETİCİ KUTUSU VE TASARIM ---
st.markdown("""
    <style>
    .stAppDeployButton {display:none !important;}
    footer {visibility: hidden !important;}
   
    div[data-testid="stExpander"] {
        position: relative !important;
        margin-bottom: 10px !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
    }

    div[data-testid="stExpander"] summary {
        padding: 8px 12px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
   
    .big-title {
        font-size: 2.2rem !important;
        font-weight: 800;
        color: #0F172A;
        margin-top: 20px;
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

# --- PDF LİSTESİ VE OTURUM YÖNETİMİ ---
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

        st.divider()
        if st.button("🗑️ Sohbet Geçmişini Sıfırla"):
            clear_db()
            st.session_state.messages = []
            st.success("Geçmiş temizlendi!")
            st.rerun()

    elif girilen_sifre != "":
        st.error("Hatalı şifre!")

# --- PDF METİN OKUMA TEST PANELİ (DEBUG) ---
with st.expander("🔍 PDF Metin Okuma Kontrolü (Debug)"):
    if mevcut_pdfler:
        secilen_test_pdf = st.selectbox("İncelemek istediğin PDF'i seç:", mevcut_pdfler)
        if secilen_test_pdf:
            try:
                reader = PdfReader(secilen_test_pdf)
                st.write(f"**Toplam Sayfa Sayısı:** {len(reader.pages)}")
               
                sayfa_no = st.number_input("Sayfa No Seç:", min_value=1, max_value=len(reader.pages), value=1)
               
                ham_metin = reader.pages[sayfa_no - 1].extract_text() or ""
               
                st.markdown(f"**{sayfa_no}. Sayfadan Çekilen Toplam Karakter Sayısı:** {len(ham_metin)}")
                st.text_area(
                    "Kütüphanenin Okuduğu Ham Metin (Aynen bu şekilde Gemini'ye gidiyor):",
                    value=ham_metin if ham_metin else "⚠️ Bu sayfadan HİÇ METİN ÇEKİLEMEDİ! (Sayfa resim veya korumalı olabilir)",
                    height=250
                )
            except Exception as e:
                st.error(f"PDF okunurken hata oluştu: {e}")
    else:
        st.info("Sistemde incelenecek PDF bulunamadı.")

# --- GELİŞMİŞ PDF CHUNKING (SLIDING WINDOW) ---
@st.cache_data
def pdf_paragraflari_cikar(pdf_listesi, chunk_size=800, overlap=150):
    paragraflar = []
    for pdf_yolu in pdf_listesi:
        try:
            reader = PdfReader(pdf_yolu)
            tum_pdf_metni = ""
            for page in reader.pages:
                metin = page.extract_text() or ""
                temiz_metin = re.sub(r'\s+', ' ', metin).strip()
                if temiz_metin:
                    tum_pdf_metni += temiz_metin + " "
            
            i = 0
            while i < len(tum_pdf_metni):
                chunk = tum_pdf_metni[i:i + chunk_size]
                if len(chunk.strip()) > 30:
                    paragraflar.append(chunk.strip())
                i += (chunk_size - overlap)
        except Exception:
            continue
    return paragraflar

# --- SORGU GENİŞLETME (QUERY EXPANSION) ---
def sorgu_genislet_havacilik(soru_metni):
    try:
        expansion_prompt = f"""Kullanıcının şu sorusundaki havacılık/ATC terimlerini İngilizce teknik karşılıklarıyla genişlet: "{soru_metni}"
Sadece arama terimleri döndür. Örnek: "radar kaybı sofya" -> "radar kaybı sofya radar failure loss of radar separation Sofia ACC"
Genişletilmiş Arama Metni:"""
        
        res = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=expansion_prompt
        )
        return f"{soru_metni} {res.text.strip()}"
    except Exception:
        return soru_metni

# --- ARAMA MOTORU (TF-IDF - GELİŞTİRİLMİŞ MİMARİ) ---
def dinamik_baglam_tfidf(soru, paragraflar, max_karakter=15000):
    if not paragraflar:
        return ""
   
    try:
        genisletilmis_soru = sorgu_genislet_havacilik(soru)
        
        tum_metinler = paragraflar + [genisletilmis_soru]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=None).fit_transform(tum_metinler)
        vectors = vectorizer.toarray()
       
        paragraf_vektorleri = vectors[:-1]
        soru_vektoru = vectors[-1:]
       
        skorlar = cosine_similarity(paragraf_vektorleri, soru_vektoru).flatten()
        skorlu_paragraflar = sorted(zip(skorlar, paragraflar), key=lambda x: x[0], reverse=True)
       
        secilen_parcalar = []
        toplam_uzunluk = 0
       
        for skor, p in skorlu_paragraflar:
            if skor > 0:
                if toplam_uzunluk + len(p) <= max_karakter:
                    secilen_parcalar.append(p)
                    toplam_uzunluk += len(p)
                if len(secilen_parcalar) >= 12:
                    break
                   
        if not secilen_parcalar and skorlu_paragraflar:
            secilen_parcalar = [p for _, p in skorlu_paragraflar[:5]]

        return "\n\n---\n\n".join(secilen_parcalar)
    except Exception as e:
        return "\n\n---\n\n".join(paragraflar[:8])

# --- BAŞLIK VE PDF YÜKLEME ---
st.markdown('<div class="big-title">Belge Asistanı  ✈️ </div>', unsafe_allow_html=True)

aktif_pdfler = st.session_state.get("secili_pdfler", [])
paragraflar = pdf_paragraflari_cikar(aktif_pdfler)

if not aktif_pdfler:
    st.warning("⚠️ Lütfen Yönetici panelinden soruların aranacağı en az 1 adet PDF seçin.")
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

    baglam = dinamik_baglam_tfidf(kullanici_sorusu, paragraflar, max_karakter=15000)

    prompt_metni = f"""
Sen "Belge Asistanı  ✈️ " adında zeki bir asistansın.

GİZLİLİK KURALI:
Sistemdeki dosya isimlerini (örneğin ACC.pdf vb.) kesinlikle açıklama. Dosya ismi sorulursa "Güvenlik nedeniyle dosya bilgilerini paylaşamıyorum." de.

TALİMATLAR:
1. Sana verilen İLGİLİ DÖKÜMAN İÇERİĞİ'ni dikkatlice incele. İçerik Türkçe veya İngilizce olabilir, kural ve sayıları buna göre değerlendir.
2. Kullanıcının sorusuna detaylı, açıklayıcı ve kaliteli bir yanıt ver.
3. Eğer metinde doğrudan yanıt varsa tam ve net bilgi ver.
4. Kullanıcının sorduğu soru verilen dökümanda HİÇ GEÇMİYORSA:
   - İlk satıra TAM OLARAK şunu yaz: "Aradığınız bilgi dökümanda doğrudan bulunamadı."
   - Ardından dökümandaki en yakın konu hakkında bilgi ver.

İLGİLİ DÖKÜMAN İÇERİĞİ:
{baglam}

KULLANICI SORUSU:
{kullanici_sorusu}
"""

    with st.chat_message("assistant"):
        with st.spinner("Belgeler inceleniyor..."):
            try:
                response = client.models.generate_content(
                    model='gemini-3.6-flash',
                    contents=prompt_metni
                )
                cevap = response.text
                st.markdown(cevap)
                st.session_state.messages.append({"role": "assistant", "content": cevap})
                save_message("assistant", cevap)
            except Exception as e:
                st.error(f"Hata oluştu: {e}")