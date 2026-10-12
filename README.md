# Aviation Document Assistant (RAG System)

A high-performance **Retrieval-Augmented Generation (RAG)** application built with Python and Streamlit. This assistant enables users to upload complex aviation documents (PDFs) and perform precise, context-aware semantic search and Question-Answering (QA) using Google's Gemini models and ChromaDB vector store.

---

## 📌 Features

* **PDF Document Ingestion:** Extracts text from uploaded aviation PDF manuals, documentation, and technical guidelines.
* **Vector Search & Embeddings:** Chunking and embedding generation using ChromaDB for high-accuracy semantic retrieval.
* **LLM-Powered QA:** Generates accurate, context-grounded answers powered by Google's Gemini LLM.
* **Source Citation:** Displays relevant context snippets and source references for transparency and verification.
* **Interactive UI:** Clean, intuitive web interface powered by Streamlit.

---

## 🏗️ Architecture

```mermaid
graph TD
    A[PDF Upload] --> B[Text Extraction & Chunking]
    B --> C[Generate Embeddings]
    C --> D[(Chroma Vector DB)]
    E[User Query] --> F[Vector Similarity Search]
    D --> F
    F --> G[Relevant Context Chunks]
    G --> H[Gemini LLM]
    E --> H
    H --> I[Grounded Answer & Citations]
🛠️ Tech Stack
Language: Python 3.10+

Frontend: Streamlit

LLM Framework / Model: Google Gemini API

Vector Database: ChromaDB

Document Processing: PyPDF / LangChain / LlamaIndex

🚀 Quick Start
1. Prerequisites
Make sure you have Python 3.10 or higher installed. You will also need a Google Gemini API Key.

2. Installation
Clone the repository and install the required dependencies:
