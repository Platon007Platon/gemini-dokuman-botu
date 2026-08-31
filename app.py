import streamlit as st
import pypdf
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from google import genai

st.set_page_config(page_title="Vektör RAG Doküman Botu", layout="wide")

# Vektörleştirme modelini önbelleğe al (Hızlı yükleme için)
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

def vektör_indeksi_olustur(parcalar):
    # Metinleri matematiksel vektörlere dönüştür
    embeddings = embedder.encode(parcalar, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    
    # FAISS vektör veritabanını oluştur ve vektörleri ekle
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

st.title("🚀 Vektör RAG (FAISS) Destekli PDF Botu")
st.write("Yüzlerce sayfalık PDF'leri yükleyin, en alakalı kısımlardan nokta atışı yanıtlar alın.")

uploaded_file = st.file_uploader("PDF Dosyanızı Yükleyin", type="pdf")

if uploaded_file:
    with st.spinner("PDF parçalanıyor ve Vektör Veritabanına (FAISS) indeksleniyor..."):
        parcalar = pdf_parcala(uploaded_file)
        index, _ = vektör_indeksi_olustur(parcalar)
        st.success(f"İşlem Tamamlandı! Toplam {len(parcalar)} vektör parçası oluşturuldu.")

    soru = st.text_input("Doküman hakkında bir soru sorun:")
    
    if soru:
        with st.spinner("Vektör alanında en alakalı paragraflar aranıyor..."):
            # Soruyu vektöre çevir ve en yakın 3 parçayı bul
            soru_vektoru = embedder.encode([soru], convert_to_numpy=True)
            distances, indices = index.search(np.array(soru_vektoru).astype('float32'), k=3)
            
            baglam = "\n---\n".join([parcalar[i] for i in indices[0] if i < len(parcalar)])
            
            prompt = f"""
            Sadece aşağıdaki doküman parçalarına dayanarak soruyu Türkçe olarak yanıtla. 
            Eğer cevap dokümanda yoksa 'Bu bilgi yüklenen dokümanda yer almıyor' de.

            DOKÜMAN PARÇALARI:
            {baglam}

            SORU: {soru}
            """
            
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            st.write("### Yanıt:")
            st.write(response.text)
            
            with st.expander("Gemini'ye Gönderilen Alakalı Vektör Parçalarını Gör"):
                st.write(baglam)