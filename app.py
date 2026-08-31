import os
import glob
import sqlite3
import streamlit as st
from google import genai
from pypdf import PdfReader

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
YEDEK_API_KEY = ""
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", YEDEK_API_KEY)
except Exception:
    GEMINI_API_KEY = YEDEK_API_KEY

if not GEMINI_API_KEY:
    st.error("⚠️ API Anahtarı bulunamadı!")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# --- PDF İŞLEME VE AKILLI PARÇALAMA ---
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

# --- AKILLI VE HAFİF PDF PARÇALAYICI ---
@st.cache_data
def pdf_paragraflari_cikar(pdf_listesi):
    """PDF'leri satir ve paragraf yapisina gore anlamli parçalara boler."""
    paragraflar = []
    for pdf_yolu in pdf_listesi:
        try:
            reader = PdfReader(pdf_yolu)
            for page in reader.pages:
                metin = page.extract_text() or ""
                # Paragraf ve blok ayrışımı
                bloklar = metin.split("\n\n")
                for b in bloklar:
                    temiz_b = b.strip().replace("\n", " ")
                    if len(temiz_b) > 30:  # Çok kısa anlamsız satırları süz
                        paragraflar.append(temiz_b)
        except Exception:
            continue
    return paragraflar

def dinamik_baglam_limitleyici(soru, paragraflar, max_karakter=4000):
    """
    Sorudaki anahtar kelimelere gore paragraflari skorlar.
    Gemini'ye gidecek toplam metni kesinlikle max_karakter sinirinda tutar.
    """
    if not paragraflar:
        return ""
        
    # Önemsiz durak kelimeleri ayıkla
    stop_words = {"ve", "veya", "ile", "de", "da", "bu", "şu", "ne", "nasıl", "neden", "için", "bir"}
    soru_kelimeleri = [k.lower() for k in soru.split() if k.lower() not in stop_words and len(k) > 2]
    
    if not soru_kelimeleri:
        soru_kelimeleri = [k.lower() for k in soru.split() if len(k) > 2]

    skorlu_list = []
    for p in paragraflar:
        p_lower = p.lower()
        # Kelime eşleşme skoru
        skor = sum(p_lower.count(k) for k in soru_kelimeleri)
        if skor > 0:
            skorlu_list.append((skor, p))
            
    # Skora göre büyükten küçüğe sırala
    skorlu_list.sort(key=lambda x: x[0], reverse=True)
    
    secilen_parcalar = []
    toplam_uzunluk = 0
    
    for skor, p in skorlu_list:
        if toplam_uzunluk + len(p) <= max_karakter:
            secilen_parcalar.append(p)
            toplam_uzunluk += len(p)
        else:
            break
            
    # Doğrudan kelime eşleşmediyse belgenin başından boyut kadar al
    if not secilen_parcalar:
        for p in paragraflar:
            if toplam_uzunluk + len(p) <= max_karakter:
                secilen_parcalar.append(p)
                toplam_uzunluk += len(p)
            else:
                break

    return "\n\n---\n\n".join(secilen_parcalar)

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

    # Maksimum 4000 karakterlik kompakt bağlam oluşturulur
    baglam = dinamik_baglam_limitleyici(kullanici_sorusu, paragraflar, max_karakter=4000)

    prompt_metni = f"""
Sen "Belge Asistanı  ✈️ " adında zeki bir asistansın.

GİZLİLİK KURALI:
Sistemdeki dosya isimlerini (örneğin ACC.pdf vb.) kesinlikle açıklama. Dosya ismi sorulursa "Güvenlik nedeniyle dosya bilgilerini paylaşamıyorum." de.

TALİMATLAR:
1. Sana verilen İLGİLİ DÖKÜMAN İÇERİĞİ'ni dikkatlice incele.
2. Kullanıcının sorusuna detaylı, açıklayıcı ve kaliteli bir yanıt ver.
3. Kullanıcının sorduğu soru verilen dökümanda HİÇ GEÇMİYORSA:
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
                # Orijinal model tanımın korundu
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