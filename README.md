# 📄 RAG-Based Aviation & Document Assistant (Belge Asistanı)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)](https://www.langchain.com/)
[![Gemini API](https://img.shields.io/badge/Gemini%20API-Embeddings%20%26%20Chat-green)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)](https://streamlit.io/)

Kullanıcıların yüklediği PDF, TXT veya dokümanlar üzerinden akıllı sorular sorulmasını, vektör tabanlı arama yapılmasını ve Gemini LLM desteğiyle kaynağa dayalı (RAG) yanıtlar alınmasını sağlayan Streamlit tabanlı doküman asistanı uygulaması.

---

## 🏗️ Sistem Mimarisi (RAG Workflow)

Sistem; yüklenen dokümanların parçalanması, Google Gemini Embedding modelleri ile vektörleştirilmesi ve ChromaDB üzerinde saklanarak kullanıcı sorgularına en yakın bağlamın LLM'e sunulması mantığıyla çalışır.

---

## 🚀 Çalıştırma Adımları (Getting Started)

### 1. Repoyu Klonlayın
git clone https://github.com/KULLANICI_ADIN/BELGE_ASISTANI_REPO_ADI.git
cd BELGE_ASISTANI_REPO_ADI

### 2. Sanal Ortam (Virtual Environment) Oluşturun
python -m venv venv
# Windows için:
venv\Scripts\activate
# macOS / Linux için:
source venv/bin/activate

### 3. Bağımlılıkları Yükleyin
pip install streamlit langchain-core langchain-google-genai langchain-chroma pypdf python-dotenv

### 4. Çevre Değişkenlerini (.env) Ayarlayın
Projenin ana dizininde bir .env dosyası oluşturun ve Gemini API anahtarınızı ekleyin:
GOOGLE_API_KEY=senin_gemini_api_anahtarin_buraya

### 5. Uygulamayı Başlatın
streamlit run app.py

---

## 🛠️ Kullanılan Teknolojiler
- Yapay Zeka & Dil Modeli: Google Gemini API (gemini-1.5-flash)
- Metin Temsili (Embeddings): Google Generative AI Embeddings (models/text-embedding-004)
- Vektör Veritabanı: ChromaDB
- Dosya Okuma: PyPDF & LangChain Document Loaders
- Arayüz: Streamlit
