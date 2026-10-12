# Aviation Document Assistant (RAG System)

A high-performance Retrieval-Augmented Generation (RAG) application built with Python and Streamlit. This assistant enables users to upload complex aviation documents (PDFs) and perform precise, context-aware semantic search and Question-Answering (QA) using Google's Gemini models and ChromaDB vector store.

## Features

* PDF Document Ingestion: Extracts text from uploaded aviation PDF manuals, documentation, and technical guidelines.
* Vector Search & Embeddings: Chunking and embedding generation using ChromaDB for high-accuracy semantic retrieval.
* LLM-Powered QA: Generates accurate, context-grounded answers powered by Google's Gemini LLM.
* Source Citation: Displays relevant context snippets and source references for transparency and verification.
* Interactive UI: Clean, intuitive web interface powered by Streamlit.

## Tech Stack

* Language: Python 3.10+
* Frontend: Streamlit
* LLM Framework / Model: Google Gemini API
* Vector Database: ChromaDB
* Document Processing: PyPDF / LangChain / LlamaIndex

## Quick Start

1. Prerequisites
Make sure you have Python 3.10 or higher installed. You will also need a Google Gemini API Key.

2. Installation
Clone the repository and install the required dependencies:
git clone [https://github.com/your-username/aviation-document-assistant.git](https://github.com/your-username/aviation-document-assistant.git)
cd aviation-document-assistant
pip install -r requirements.txt

3. Environment Setup
Create a .env file in the root directory and add your Google Gemini API key:
GEMINI_API_KEY=your_actual_gemini_api_key_here

4. Running the Application
Launch the Streamlit server:
streamlit run app.py

Open your browser and navigate to http://localhost:8501.

## Project Structure

├── app.py                # Main Streamlit application entry point
├── requirements.txt      # Python dependencies
├── .env.example          # Sample environment configuration file
├── README.md             # Project documentation
└── src/                  # Source code directory
    ├── ingest.py         # PDF processing and chunking logic
    ├── vector_store.py   # ChromaDB integration & embeddings
    └── rag_chain.py      # LLM prompt and query pipeline

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for any enhancements or bug fixes.

