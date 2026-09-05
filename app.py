import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import sqlite3
import re
import time
import random
import pickle
import hashlib
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
import pdfplumber
from pdf2image import convert_from_path
from PIL import ImageEnhance

# --- LANGCHAIN IMPORTLARI ---
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# --- HİBRİT RETRIEVER ---
class CustomEnsembleRetriever:
    """BM25 ve FAISS sonuçlarını birleştiren ve skora göre sıralayan hibrit retriever."""
    def __init__(self, bm25, faiss, k=15):
        self.bm25 = bm25
        self.faiss = faiss
        self.k = k

    def invoke(self, query: str):
        bm25_docs = self.bm25.invoke(query)
        faiss_docs = self.faiss.invoke(query)

        doc_scores = {}
        doc_map = {}

        for rank, doc in enumerate(bm25_docs):
            content = doc.page_content
            doc_scores[content] = doc_scores.get(content, 0) + (1.0 / (rank + 60))
            doc_map[content] = doc

        for rank, doc in enumerate(faiss_docs):
            content = doc.page_content
            doc_scores[content] = doc_scores.get(content, 0) + (1.0 / (rank + 60))
            doc_map[content] = doc

        sorted_contents = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

        return [doc_map[c] for c in sorted_contents[:self.k]]


# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="Belge Asistanı ✈️",
    page_icon="✈️",
    layout="centered"
)

# --- 2. VERİTABANI İŞLEMLERİ ---
DB_FILE = "chat_history.db"

def get_db_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False, timeout=10)

def init_db():
    with get_db_connection() as conn:
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

def save_message(role: str, content: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
        conn.commit()

def load_messages():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM messages ORDER BY id ASC")
        rows = cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]

def clear_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")
        conn.commit()

init_db()

# DİSK ÖNBELLEK KLASÖRLERİ
# Mutlak yol kullanılıyor: uygulama farklı bir çalışma dizininden (cwd) başlatılsa bile
# önbellek hep aynı klasörde kalır, her seferinde boş klasör oluşup yeniden okuma yapılmaz.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "pdf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

INDEX_CACHE_DIR = os.path.join(BASE_DIR, "index_cache")
os.makedirs(INDEX_CACHE_DIR, exist_ok=True)


def _index_onbellek_anahtari(pdf_listesi):
    """Seçili PDF listesinin İÇERİĞİNE göre benzersiz bir anahtar üretir (mtime DEĞİL).
    Neden içerik hash'i: Git'e commit edilip GitHub üzerinden Streamlit Cloud'a
    çekildiğinde dosyaların değişiklik zamanı (mtime) checkout anına göre sıfırlanır.
    mtime kullansaydık her deploy'da anahtar değişir, önbellek hiç eşleşmezdi.
    İçerik hash'i ise dosya gerçekten değişmediği sürece her ortamda aynı kalır."""
    hasher = hashlib.md5()
    for pdf in sorted(pdf_listesi):
        hasher.update(os.path.basename(pdf).encode("utf-8"))
        try:
            with open(pdf, "rb") as f:
                hasher.update(f.read())
        except OSError:
            pass
    return hasher.hexdigest()

# --- CSS: TASARIM ---
st.markdown("""
    <style>
    .stAppDeployButton {display:none !important;}
    footer {visibility: hidden !important;}

    .big-title {
        font-size: 2.2rem !important;
        font-weight: 800;
        color: #0F172A;
        margin-top: 10px;
        margin-bottom: 2px;
    }
    .sub-caption {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- GÜVENLİK KONTROLÜ (API KEY & YÖNETİCİ ŞİFRESİ) ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "1234")  # Tanımlı değilse varsayılan: 1234

if not GEMINI_API_KEY:
    st.error("⚠️ Gemini API Anahtarı (.streamlit/secrets.toml veya Cloud Secrets) bulunamadı!")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# --- MODEL AYARLARI ---
BIRINCIL_MODEL = "gemini-3.6-flash"
YEDEK_MODEL = "gemini-3.5-flash-lite"


def gemini_cagir_guvenli(prompt_or_contents, max_deneme: int = 5, sicaklik: float = None) -> str:
    """Kotaya takılmamak için üssel bekleme (exponential backoff) içeren güvenli API çağırıcı.

    sicaklik (temperature) parametresi verilirse deterministik/tutarlı çıktı için
    config olarak API'ye iletilir. Tablo çıkarımı gibi hassas görevlerde 0.0 önerilir.
    """
    modeller = [BIRINCIL_MODEL, YEDEK_MODEL]
    son_hata = None

    config = None
    if sicaklik is not None:
        config = genai_types.GenerateContentConfig(temperature=sicaklik)

    for model_adi in modeller:
        for deneme in range(max_deneme):
            try:
                response = client.models.generate_content(
                    model=model_adi,
                    contents=prompt_or_contents,
                    config=config,
                )
                return response.text
            except genai_errors.ServerError as e:
                son_hata = e
                bekleme = (2 ** deneme) + random.uniform(1, 3)
                time.sleep(bekleme)
                continue
            except genai_errors.ClientError as e:
                son_hata = e
                if getattr(e, "code", None) == 429 or "429" in str(e):
                    bekleme = (4 * (deneme + 1)) + random.uniform(1, 2)
                    time.sleep(bekleme)
                    continue
                raise e

    raise son_hata


def gemini_vision_ile_pdf_oku(pdf_yolu):
    """
    Disk önbelleği destekli Vision okuma fonksiyonu.
    Okunan içerikler pdf_cache/ dizininde .txt olarak kaydedilir.
    Sonraki çalıştırmalarda 0 API kotası harcar.

    Tablo çıkarımı doğruluğunu artırmak için:
    - Yüksek DPI (250) ile sayfa görüntüsü alınır
    - Kontrast ve keskinlik artırılır (soluk/taranmış PDF'ler için)
    - Prompt, tabloların satır/sütun bütünlüğünü koruyacak şekilde katılaştırılmıştır
    - temperature=0.0 ile deterministik/tutarlı çıktı sağlanır
    """
    pdf_adi = os.path.basename(pdf_yolu)
    cache_dosya_yolu = os.path.join(CACHE_DIR, f"{pdf_adi}.txt")

    # 1. ÖNCELİK: Daha önce okunup kaydedildiyse diskten getir (Sıfır API Harcaması)
    if os.path.exists(cache_dosya_yolu):
        try:
            with open(cache_dosya_yolu, "r", encoding="utf-8") as f:
                kayitli_metin = f.read()
            if kayitli_metin.strip():
                return kayitli_metin, None, "cache"
        except Exception:
            pass

    # 2. ÖNCELİK: Önbellekte yoksa Gemini Vision ile sayfa sayfa oku
    try:
        # DPI artırıldı: 150 -> 250. Küçük puntolu tablo hücreleri (FL seviyeleri,
        # koordinatlar, kod tabloları) için okunabilirliği ciddi şekilde iyileştirir.
        images = convert_from_path(pdf_yolu, dpi=250)
        vision_metin = ""

        prompt = (
            "Bu görsel bir havacılık dökümanına/tablosuna aittir. Aşağıdaki adımları SIRAYLA uygula:\n\n"
            "1) ÖNCE görselde kaç tablo olduğunu ve her tablonun kaç sütun/satırdan oluştuğunu belirle "
            "(bunu çıktıya yazma, sadece kendi içinde planla).\n\n"
            "2) Görseldeki TÜM metni, başlıkları, havalimanı kodlarını ve FL seviyelerini eksiksiz çıkar.\n\n"
            "3) TABLOLAR İÇİN KRİTİK KURALLAR:\n"
            "   - Her tabloyu düzgün Markdown tablo formatında yaz: | Sütun1 | Sütun2 | ...\n"
            "   - Başlık satırındaki sütun sayısı ile TÜM veri satırlarındaki hücre sayısı BİREBİR AYNI "
            "olmalı. Eksik hücre varsa boş bırakma, '-' koy, ama satırı ASLA atlama.\n"
            "   - Birleştirilmiş (merged) hücreler varsa değeri her ilgili sütuna da tekrar yaz, boş bırakma.\n"
            "   - Görselde kaç satır veri görüyorsan çıktıda da tam o kadar satır olmalı. Satır atlamak veya "
            "özetlemek KESİNLİKLE YASAK.\n"
            "   - Çok küçük/soluk yazılmış rakamları da dikkatlice oku; emin değilsen en yakın tahmini yaz "
            "ve başına [?] işareti koy, ama boş bırakma.\n"
            "   - Tablo birden fazla bölüme ayrılmışsa (örneğin sayfa ortasında ikinci bir tablo varsa) "
            "her birini ayrı Markdown tablosu olarak, aralarına '---' koyarak yaz.\n\n"
            "4) Tablo dışındaki serbest metni tablo formatına sokmadan normal paragraf olarak yaz.\n\n"
            "5) Çıktının sonuna 'TOPLAM SATIR SAYISI: N' şeklinde her tablo için gördüğün veri satırı "
            "sayısını yaz (kontrol amaçlı, doğrulama için)."
        )

        toplam_sayfa = len(images)
        progress_bar = st.progress(0, text=f"{pdf_adi} Vision ile işleniyor (0/{toplam_sayfa})...")

        for idx, img in enumerate(images):
            # Kontrast ve keskinliği artırarak soluk/düşük kaliteli taramalarda
            # tablo çizgileri ve rakamların ayrışmasını iyileştir.
            img_islenmis = img.convert("RGB")
            img_islenmis = ImageEnhance.Contrast(img_islenmis).enhance(1.4)
            img_islenmis = ImageEnhance.Sharpness(img_islenmis).enhance(1.5)

            sayfa_cevabi = gemini_cagir_guvenli(
                [img_islenmis, prompt],
                sicaklik=0.0,
            )

            if sayfa_cevabi:
                vision_metin += f"\n--- SAYFA {idx+1} ({pdf_adi}) ---\n" + sayfa_cevabi + "\n"

            # Rate limit (15 RPM) sınırı için sayfa arası gecikme
            time.sleep(2.5)
            progress_bar.progress((idx + 1) / toplam_sayfa, text=f"{pdf_adi} Vision ile işleniyor ({idx+1}/{toplam_sayfa})...")

        progress_bar.empty()

        if not vision_metin.strip():
            return "", "Gemini Vision metin üretemedi.", "api"

        # 3. BAŞARILI OKUMAYI GELECEK ÇALIŞTIRMALAR İÇİN DİSKE KAYDET
        with open(cache_dosya_yolu, "w", encoding="utf-8") as f:
            f.write(vision_metin)

        return vision_metin, None, "api"
    except Exception as e:
        return "", f"Gemini Vision okuma hatası: {e}", "api"


# --- PDF LİSTESİ VE OTURUM YÖNETİMİ ---
mevcut_pdfler = glob.glob("*.pdf")

if "secili_pdfler" not in st.session_state:
    st.session_state.secili_pdfler = mevcut_pdfler.copy()

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

# --- SIDEBAR (ŞİFRE KORUMALI YÖNETİCİ PANELİ) ---
with st.sidebar:
    st.header("⚙️ Yönetim Paneli")

    if not st.session_state.admin_logged_in:
        st.subheader("🔒 Yönetici Girişi")
        girilen_sifre = st.text_input("Yönetici Şifresi", type="password", key="admin_password_input")

        if st.button("Giriş Yap", use_container_width=True):
            if girilen_sifre == ADMIN_PASSWORD:
                st.session_state.admin_logged_in = True
                st.success("Yönetici girişi başarılı!")
                st.rerun()
            else:
                st.error("❌ Hatalı Şifre!")
        st.info("💡 Yönetici paneline erişmek için şifre giriniz.")
    else:
        st.success("🔓 Yönetici Oturumu Açık")
        if st.button("🚪 Çıkış Yap", use_container_width=True):
            st.session_state.admin_logged_in = False
            st.rerun()

        st.divider()

        if st.button("🗑️ Sohbet Geçmişini Sıfırla", use_container_width=True):
            clear_db()
            st.session_state.messages = []
            st.cache_resource.clear()
            st.success("Geçmiş ve hafıza temizlendi!")
            st.rerun()

        st.divider()

        st.subheader("🔧 Yönetici Ayarları")
        yuklenen_dosya = st.file_uploader("Yeni PDF ekle", type=["pdf"])
        if yuklenen_dosya:
            dosya_adi = yuklenen_dosya.name
            with open(dosya_adi, "wb") as f:
                f.write(yuklenen_dosya.getbuffer())

            st.cache_resource.clear()

            if dosya_adi not in st.session_state.secili_pdfler:
                st.session_state.secili_pdfler.append(dosya_adi)
            st.success(f"'{dosya_adi}' yüklendi!")
            st.rerun()

        st.divider()
        st.markdown("**Aktif Edilecek Belgeler:**")

        secilen_liste = []
        for pdf in mevcut_pdfler:
            varsayilan = pdf in st.session_state.secili_pdfler
            if st.checkbox(pdf, value=varsayilan, key=f"chk_{pdf}"):
                secilen_liste.append(pdf)

        if secilen_liste != st.session_state.secili_pdfler:
            st.session_state.secili_pdfler = secilen_liste
            st.cache_resource.clear()

        st.divider()
        st.markdown("**📋 Son Okuma Tanı Raporu:**")
        if "son_okuma_raporu" in st.session_state and st.session_state.son_okuma_raporu:
            for dosya_adi, durum, detay in st.session_state.son_okuma_raporu:
                if durum == "ok":
                    st.success(f"✅ {dosya_adi}: {detay}")
                elif durum == "vision_cache":
                    st.info(f"💾 {dosya_adi}: Vision (önbellekten, API çağrısı YOK) — {detay}")
                elif durum == "vision_api":
                    st.warning(f"👁️ {dosya_adi}: Vision (YENİ API çağrısı yapıldı) — {detay}")
                else:
                    st.error(f"❌ {dosya_adi}: {detay}")


@st.cache_resource
def hibrit_arama_sistemi_kur(pdf_listesi):
    if not pdf_listesi:
        return None, None, []

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    onbellek_anahtari = _index_onbellek_anahtari(pdf_listesi)
    faiss_yolu = os.path.join(INDEX_CACHE_DIR, f"faiss_{onbellek_anahtari}")
    yan_veri_yolu = os.path.join(INDEX_CACHE_DIR, f"yan_veri_{onbellek_anahtari}.pkl")

    # DİSK ÖNBELLEĞİNDEN HIZLI YÜKLEME: Bu PDF seti için FAISS index + BM25 + rapor
    # daha önce kurulup diske kaydedildiyse, PDF okuma/Vision/embedding hesaplama
    # adımlarının TAMAMINI atlayıp doğrudan diskten yükle. Uygulama her yeniden
    # başlatıldığında yavaşlığın asıl kaynağı genelde budur (Vision cache'i olsa bile).
    if os.path.exists(faiss_yolu) and os.path.exists(yan_veri_yolu):
        try:
            vectorstore = FAISS.load_local(
                faiss_yolu, embeddings, allow_dangerous_deserialization=True
            )
            with open(yan_veri_yolu, "rb") as f:
                yan_veri = pickle.load(f)

            faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
            ensemble_retriever = CustomEnsembleRetriever(
                bm25=yan_veri["bm25"], faiss=faiss_retriever, k=20
            )
            return ensemble_retriever, yan_veri["splits"], yan_veri["okuma_raporu"]
        except Exception:
            pass  # Bozuk/uyumsuz önbellek varsa sessizce sıfırdan kur

    raw_documents = []
    okuma_raporu = []

    for pdf_yolu in pdf_listesi:
        tum_metin = ""
        try:
            with pdfplumber.open(pdf_yolu) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    metin = page.extract_text(layout=True) or page.extract_text(layout=False) or ""
                    temiz_metin = re.sub(r'[ \t]+', ' ', metin).strip()
                    if temiz_metin:
                        tum_metin += f"\n--- SAYFA {page_num} ---\n" + temiz_metin + "\n"
        except Exception:
            tum_metin = ""

        # Dijital metin tabanlı PDF ise doğrudan ekle
        if tum_metin.strip() and len(tum_metin.strip()) > 100:
            raw_documents.append(Document(page_content=tum_metin, metadata={"source": str(pdf_yolu)}))
            okuma_raporu.append((pdf_yolu, "ok", f"{len(tum_metin.strip())} karakter (Metin Okuma)"))
            continue

        # Taranmış veya görsel ağırlıklı PDF ise Vision (Önbellek destekli) çalıştır
        vision_metin, vision_hata, vision_kaynak = gemini_vision_ile_pdf_oku(pdf_yolu)

        if vision_metin.strip():
            raw_documents.append(Document(page_content=vision_metin, metadata={"source": str(pdf_yolu)}))
            durum = "vision_cache" if vision_kaynak == "cache" else "vision_api"
            okuma_raporu.append((pdf_yolu, durum, f"{len(vision_metin.strip())} karakter (Gemini Vision)"))
        else:
            okuma_raporu.append((pdf_yolu, "basarisiz", vision_hata or "Okunamadı"))

    if not raw_documents:
        return None, None, okuma_raporu

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    splits = text_splitter.split_documents(raw_documents)

    vectorstore = FAISS.from_documents(splits, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 15

    ensemble_retriever = CustomEnsembleRetriever(
        bm25=bm25_retriever,
        faiss=faiss_retriever,
        k=20
    )

    # DİSK ÖNBELLEĞİNE KAYDET: Bir sonraki uygulama açılışında embedding/BM25
    # yeniden hesaplanmasın diye FAISS index'i ve yan verileri diske yaz.
    try:
        vectorstore.save_local(faiss_yolu)
        with open(yan_veri_yolu, "wb") as f:
            pickle.dump(
                {"bm25": bm25_retriever, "okuma_raporu": okuma_raporu, "splits": splits},
                f,
            )
    except Exception:
        pass  # Önbelleğe yazma başarısız olsa bile uygulama çalışmaya devam etsin

    return ensemble_retriever, splits, okuma_raporu


def havacilik_arama_genislet(soru):
    eslesmeler = {
        "sinop": "LTSB SINOP Sinop arrival landing level FL alçalma yaklaşma LTAA mutabakat muhtırası tablo mutabakat",
        "merzifon": "LTAP MERZIFON Merzifon arrival landing level FL",
        "çaycuma": "LTCU CAYCUMA Zonguldak TWR arrival landing level FL",
        "kastamonu": "LTAL KASTAMONU Kastamonu landing arrival alçalma level FL",
        "esenboğa": "LTAC ESENBOGA Esenboga landing arrival level FL",
        "iniş": "landing arrival inbound descent level FL alçalma",
        "kalkış": "departure outbound climb level FL tırmanma"
    }

    ek_kelimeler = []
    soru_lower = soru.lower().replace("ı", "i").replace("İ", "i")

    for anahtar, deger in eslesmeler.items():
        if anahtar in soru_lower:
            ek_kelimeler.append(deger)

    return f"{soru} {' '.join(ek_kelimeler)}"


def hibrit_arama_yap(soru, retriever_obj):
    if not retriever_obj:
        return ""

    arama_metni = havacilik_arama_genislet(soru)
    docs = retriever_obj.invoke(arama_metni)

    benzersiz_metinler = []
    for d in docs:
        kaynak_bilgisi = d.metadata.get("source", "Bilinmeyen Kaynak")
        metin_blogu = f"[KAYNAK BELGE: {kaynak_bilgisi}]\n{d.page_content}"

        if metin_blogu not in benzersiz_metinler:
            benzersiz_metinler.append(metin_blogu)

    # retriever zaten en fazla retriever_obj.k sonuç döndürüyor; burada tekrar
    # sabit bir sayıyla kesmek yerine aynı sınırı kullanıyoruz (tek yerden yönetim).
    return "\n\n---\n\n".join(benzersiz_metinler[:retriever_obj.k])


# --- ANA EKRAN ---
st.markdown('<div class="big-title">Belge Asistanı ✈️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">Made by Serd@R T.</div>', unsafe_allow_html=True)

aktif_pdfler = st.session_state.get("secili_pdfler", [])
retriever = None

if not aktif_pdfler:
    st.warning("⚠️ Seçili belge bulunamadı. Yönetici panelinden belge aktifleştirin.")
else:
    with st.spinner("Hibrit arama motoru veritabanı hazırlanıyor..."):
        retriever, _, okuma_raporu = hibrit_arama_sistemi_kur(tuple(aktif_pdfler))
        st.session_state.son_okuma_raporu = okuma_raporu

# --- SOHBET GEÇMİŞİ ---
if "messages" not in st.session_state:
    st.session_state.messages = load_messages()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- SOHBET GİRDİSİ ---
if kullanici_sorusu := st.chat_input("Ne öğrenmek istiyorsun?"):

    st.chat_message("user").markdown(kullanici_sorusu)
    st.session_state.messages.append({"role": "user", "content": kullanici_sorusu})
    save_message("user", kullanici_sorusu)

    if retriever is None:
        st.error("⚠️ Vektör veritabanı hazırlanamadı. Lütfen geçerli bir PDF seçili olduğundan emin olun.")
    else:
        baglam = hibrit_arama_yap(kullanici_sorusu, retriever)

        prompt_metni = f"""
Sen "Belge Asistanı ✈️" adında uzman bir havacılık ve döküman asistansın.

TALİMATLAR:
1. Sana verilen İLGİLİ DÖKÜMAN İÇERİĞİ'ni dikkatlice incele. İçerikteki tablolar Markdown formatında düzenlenmiştir.
2. Kullanıcının sorusuna dökümandaki irtifa, koordinat, FL seviyeleri ve tablodaki satır/sütun eşleşmelerine dayanarak yanıt ver.
3. Soruya yanıt verirken tablolardaki sayısal değerleri doğrudan yansıt.
4. YÖN KONTROLÜ ZORUNLU: Soru bir kalkış-varış yönü içeriyorsa (örneğin "X'ten Y'ye"), bağlamdaki
   TÜM tabloları tek tek kontrol et. Aynı iki nokta arasında BİRDEN FAZLA tablo olabilir
   (örneğin biri "X'ten Y'ye", diğeri "Y'den X'e" yönünde). Sadece ilk bulduğun tabloya bakıp
   "bilgi yok" SONUCUNA VARMA — bağlamdaki tüm tabloları/bölümleri taradığından emin olmadan
   asla "bulunamadı" deme.
5. Eğer bağlamda birden fazla tablo varsa ve hangisinin soruyla eşleştiğinden emin değilsen,
   her tablonun başlığını/yön bilgisini kontrol ederek doğru olanı seç.
6. Sadece verilen dökümanda kesinlikle geçmeyen konular için, TÜM tabloları taradıktan sonra
   bilgi bulunamadığını söyle.

İLGİLİ DÖKÜMAN İÇERİĞİ:
{baglam}

KULLANICI SORUSU:
{kullanici_sorusu}
"""

        with st.chat_message("assistant"):
            with st.spinner("Belgeler inceleniyor..."):
                try:
                    cevap = gemini_cagir_guvenli(prompt_metni)
                    st.markdown(cevap)
                    st.session_state.messages.append({"role": "assistant", "content": cevap})
                    save_message("assistant", cevap)
                except Exception as e:
                    st.error(f"⚠️ Hata oluştu: {e}")