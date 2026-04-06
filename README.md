<div align="center">

# 🔍 RAG System

**A fully local Retrieval-Augmented Generation system — upload PDFs, ask questions, get grounded answers.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-orange?style=for-the-badge)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-llama3-black?style=for-the-badge)](https://ollama.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
  - [1. Populate the Vector Database](#1-populate-the-vector-database)
  - [2. Launch the Web UI](#2-launch-the-web-ui)
  - [3. Query via CLI](#3-query-via-cli)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Performance Profiling](#-performance-profiling)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

**RAG System** is a fully local, privacy-first question-answering application built on the **Retrieval-Augmented Generation (RAG)** pattern. It allows you to:

1. **Ingest** any number of PDF documents into a persistent local vector store.
2. **Ask** natural-language questions against those documents.
3. **Receive** grounded, citation-backed answers — powered entirely by locally-running models (no API keys, no data leaving your machine).

> All embeddings are generated on-device with [Sentence Transformers](https://www.sbert.net/) and all language generation is handled by [Ollama](https://ollama.ai/) running **llama3** locally.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG System Pipeline                         │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  PDF     │───▶│  Document    │───▶│  RecursiveCharacter      │  │
│  │  Files   │    │  Loader      │    │  TextSplitter            │  │
│  │ (data/)  │    │ (PyPDF)      │    │  (chunk_size=800,        │  │
│  └──────────┘    └──────────────┘    │   overlap=80)            │  │
│                                      └────────────┬─────────────┘  │
│                                                   │                 │
│                                                   ▼                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │         LocalEmbeddings (sentence-transformers/all-mpnet-v2) │  │
│  └────────────────────────────┬─────────────────────────────────┘  │
│                               │                                     │
│                               ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                  ChromaDB (chroma/)                        │    │
│  │              Persistent Local Vector Store                 │    │
│  └────────────────────────────┬───────────────────────────────┘    │
│                               │  Similarity Search (top-5)         │
│                               ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │         LangChain ChatPromptTemplate + Context Assembly    │    │
│  └────────────────────────────┬───────────────────────────────┘    │
│                               │                                     │
│                               ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              Ollama  (llama3:latest)  — local LLM          │    │
│  └────────────────────────────┬───────────────────────────────┘    │
│                               │                                     │
│                               ▼                                     │
│              Grounded Answer  +  Source Citations                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📂 **PDF Ingestion** | Upload and index unlimited PDF documents via the web UI or CLI |
| 🔒 **100% Local** | No cloud APIs — embeddings and LLM inference run entirely on your machine |
| ⚡ **Incremental Indexing** | Only new documents are embedded and added; existing chunks are never re-processed |
| 🗂 **Persistent Vector Store** | ChromaDB persists your index to disk — survives restarts |
| 🖥 **Streamlit Web UI** | Clean, interactive interface for upload, query, and results |
| 🔗 **Source Citations** | Every answer includes the source document IDs used to generate it |
| 🧪 **LLM-as-Judge Tests** | Automated evaluation suite that uses the LLM to validate answer correctness |
| 📊 **Performance Profiling** | Built-in `cProfile` integration to profile query latency |
| 🗑 **Database Reset** | One-flag CLI command to wipe and rebuild the vector store |

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**
- **[Ollama](https://ollama.ai/)** — running locally with the `llama3` model pulled:
  ```bash
  ollama pull llama3
  ```
- **Git**

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/RaptorBlingx/RAG_System.git
cd RAG_System
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the `all-mpnet-base-v2` model (~420 MB) on first run.

### 4. Create the data directory

```bash
mkdir -p data
```

---

## 🖥 Usage

### 1. Populate the Vector Database

Place your PDF files in the `data/` directory, then run:

```bash
python -m backend.database.populate_database
```

To **reset** the database and re-index from scratch:

```bash
python -m backend.database.populate_database --reset
```

### 2. Launch the Web UI

```bash
streamlit run frontend/app.py
```

Open your browser at `http://localhost:8501`. From there you can:
- Upload PDF documents directly through the interface.
- Type a natural-language question and click **Submit**.
- See the grounded answer and the response time.

### 3. Query via CLI

```bash
python -m backend.query.query_data "What are the rules for collecting rent in Monopoly?"
```

---

## 📁 Project Structure

```
RAG_System/
├── backend/
│   ├── __init__.py
│   ├── database/
│   │   └── populate_database.py   # PDF loading, chunking, and ChromaDB indexing
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── get_embedding_function.py  # Local SentenceTransformer embeddings
│   ├── query/
│   │   └── query_data.py          # RAG query pipeline (embed → retrieve → generate)
│   └── test/
│       └── test_rag.py            # LLM-as-judge evaluation tests
├── frontend/
│   ├── __init__.py
│   ├── app.py                     # Streamlit web application entry point
│   └── components/
│       ├── __init__.py
│       ├── uploader.py            # PDF upload component
│       ├── query_input.py         # Query input component
│       └── results_display.py     # Results display component
├── chroma/                        # Persisted ChromaDB vector store (auto-generated)
├── data/                          # Place your PDF documents here (you must create this)
├── config.py                      # Shared configuration constants
├── profile_query.py               # cProfile-based query performance profiler
├── requirements.txt               # Python dependencies
└── README.md
```

---

## ⚙️ Configuration

Key constants are defined directly in each module. The primary ones to know:

| Constant | Location | Default | Description |
|----------|----------|---------|-------------|
| `CHROMA_PATH` | `populate_database.py`, `query_data.py` | `"chroma"` | Path to ChromaDB persistence directory |
| `DATA_PATH` | `populate_database.py` | `"data"` | Directory scanned for PDF files |
| `chunk_size` | `populate_database.py` | `800` | Maximum characters per document chunk |
| `chunk_overlap` | `populate_database.py` | `80` | Overlap between consecutive chunks |
| `k` (top-k retrieval) | `query_data.py` | `5` | Number of similar chunks retrieved per query |
| `model_name` (embedding) | `get_embedding_function.py` | `sentence-transformers/all-mpnet-base-v2` | Local embedding model |
| Ollama model | `query_data.py` | `llama3:latest` | Local LLM used for answer generation |

---

## 🧪 Testing

The test suite uses an **LLM-as-judge** evaluation strategy — the same Ollama model evaluates whether the RAG system's answer matches the expected answer.

Run the tests with:

```bash
pytest backend/test/test_rag.py -v
```

> **Prerequisites:** Ollama must be running and the vector database must be populated with the relevant PDFs (e.g., Monopoly rules, Ticket to Ride rules) before running the tests.

Current test cases:

| Test | Question | Expected |
|------|----------|----------|
| `test_monopoly_rules` | Starting money in Monopoly? | `$1500` |
| `test_ticket_to_ride_rules` | Longest train bonus points? | `10 points` |

---

## 📊 Performance Profiling

To profile query execution and identify bottlenecks:

```bash
python profile_query.py
```

This runs `cProfile` on the full RAG pipeline and writes a sorted call-stack report to `profile_output.txt`. Inspect it with:

```bash
python -c "import pstats; pstats.Stats('profile_output.txt').sort_stats('cumulative').print_stats(20)"
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| **Web Framework** | [Streamlit](https://streamlit.io/) |
| **LLM Orchestration** | [LangChain](https://www.langchain.com/) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) |
| **Embeddings** | [Sentence Transformers](https://www.sbert.net/) — `all-mpnet-base-v2` |
| **Language Model** | [Ollama](https://ollama.ai/) — `llama3:latest` |
| **PDF Parsing** | [PyPDF2](https://pypdf2.readthedocs.io/) via LangChain |
| **Testing** | [pytest](https://docs.pytest.org/) + LLM-as-Judge |
| **Profiling** | Python `cProfile` + `pstats` |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make** your changes and ensure existing tests pass
4. **Commit** your changes:
   ```bash
   git commit -m "feat: add your feature description"
   ```
5. **Push** to your fork and open a **Pull Request**

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  Built with ❤️ using LangChain, ChromaDB, and Ollama
</div>
