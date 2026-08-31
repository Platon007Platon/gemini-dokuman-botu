import streamlit as st
import pypdf
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from google import genai

# Sayfa Yapılandırması (Orijinal Başlık ve Emoji)
st.set_page_config(page_title="Belge Asistanı ✈️", page_icon="✈️", layout="wide")

# Vektör Modelini Önbelleğe Al
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

def pdf_parcala(pdf_dosyasi, chunk_size=700, overlap=100):
    reader = pypdf.PdfReader(pdf_dosyasi)
    tam_metin = ""
    for page in reader.pages:
        metin = page.extract_text()
        if metin:
            tam_metin += metin + "\n"
    
    parcalar = []
    for i in range(0, len(tam_metin), chunk_size - overlap):
        parcalar.append(tam_metin[i:i + chunk_size])
    return parcalar

def vektor_indeksi_olustur(parcalar):
    embeddings = embedder.encode(parcalar, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

# --- ARAYÜZ (Orijinal Tasarım) ---
st.title("Belge Asistanı ✈️")

# Sol Yan Menü (Sidebar) - Şifreli Yönetici Paneli
with st.sidebar:
    st.header("Yönetici Paneli 🔒")
    sifre = st.text_input("Yönetici Şifresi:", type="password")
    
    # Varsayılan Şifre Kontrolü (İstersen değiştirebilirsin)
    YONETICI_SIFRESI = "1234"
    
    if sifre == YONETICI_SIFRESI:
        st.success("Yönetici girişi başarılı!")
        uploaded_file = st.file_uploader("Bir PDF belgesi yükleyin", type="pdf")
        if uploaded_file:
            st.session_state["aktif_pdf"] = uploaded_file
            st.success("Belge sisteme yüklendi!")
    elif sifre != "":
        st.error("Hatalı şifre!")

# Ana Ekran
if "aktif_pdf" in st.session_state and st.session_state["aktif_pdf"] is not None:
    uploaded_file = st.session_state["aktif_pdf"]
    
    # Arka planda Vektör RAG hazırlığı
    with st.spinner("Belge indeksleniyor ve vektör veritabanı hazırlanıyor..."):
        parcalar = pdf_parcala(uploaded_file)
        index = vektor_indeksi_olustur(parcalar)

    # Soru Sorma Kutusu
    soru = st.text_input("Belgenizle ilgili sorunuzu yazın:")
    
    if soru:
        with st.spinner("Yanıt hazırlanıyor..."):
            # Vektör arama ile en alakalı parçaları çek
            soru_vektoru = embedder.encode([soru], convert_to_numpy=True)
            distances, indices = index.search(np.array(soru_vektoru).astype('float32'), k=3)
            baglam = "\n---\n".join([parcalar[i] for i in indices[0] if i < len(parcalar)])
            
            prompt = f"""
            Aşağıdaki belge içeriğini dikkate alarak kullanıcı sorusunu yanıtla.
            Eğer cevap belgede yoksa, dürüstçe belgede bulunmadığını belirt.

            BELGE İÇERİĞİ:
            {baglam}

            SORU: {soru}
            """
            
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            st.markdown("### Yanıt")
            st.write(response.text)
else:
    st.info("Soru sorabilmek için sol menüden Yönetici Paneline giriş yapıp bir PDF belgesi yüklemeniz gerekmektedir.")

# Orijinal İmza (Dipnot)
st.markdown("---")
st.caption("Made by Serd@R T.")