# 🧠 RAG AI Assistant

A **production-ready** Retrieval-Augmented Generation (RAG) chatbot built with **LangChain**, **OpenAI**, **Pinecone**, and **Streamlit**. Upload PDFs, ask questions, and get AI-powered answers with source citations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4.1--mini-black?logo=openai)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-purple)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Multi-PDF Upload | Upload and process multiple PDFs simultaneously |
| 🔍 Semantic Search | Find relevant information using vector similarity |
| 💬 Conversational Chat | Follow-up questions with memory-aware context |
| 🌊 Streaming Responses | Real-time token-by-token AI responses |
| 📚 Source Citations | Every answer includes document references |
| 🎨 Dark Modern UI | Premium Streamlit interface with custom theming |
| 🐳 Docker Ready | One-command deployment with Docker |
| 🔒 Secure | API keys managed via environment variables |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────┐
│  Condense Step   │  ← Reformulates follow-ups using chat history
│  (GPT-4.1-mini)  │
└────────┬────────┘
         │  standalone question
         ▼
┌─────────────────┐
│  Pinecone Search │  ← Converts question to embedding → ANN search
│  (top-k = 5)     │
└────────┬────────┘
         │  relevant chunks
         ▼
┌─────────────────┐
│  Answer Generation│  ← Context + Question → Streaming response
│  (GPT-4.1-mini)   │
└────────┬────────┘
         │
         ▼
   Chat UI + Source Citations
```

---

## 📁 Project Structure

```
project/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── Dockerfile              # Docker configuration
├── .gitignore
├── pyproject.toml
│
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
│
├── rag/                    # Core RAG modules
│   ├── __init__.py
│   ├── ingest.py           # PDF → Chunks → Embeddings → Pinecone
│   ├── retriever.py        # Semantic search & document retrieval
│   ├── chat.py             # RAG chain with streaming
│   ├── pinecone_db.py      # Pinecone client & vector store
│   ├── memory.py           # Conversation memory management
│   └── prompts.py          # Prompt templates
│
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── pdf_loader.py       # PDF text extraction
│   ├── chunking.py         # Text splitting
│   └── helpers.py          # Environment & formatting utilities
│
└── data/                   # Uploaded files (gitignored)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **OpenAI API Key** → [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Pinecone API Key** → [app.pinecone.io](https://app.pinecone.io)

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/rag-ai-assistant.git
cd rag-ai-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your actual keys:

```env
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=rag-assistant
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

### 3. Run the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t rag-assistant .

# Run the container
docker run -p 8501:8501 --env-file .env rag-assistant
```

---

## ☁️ Cloud Deployment

### Streamlit Cloud

1. Push your code to **GitHub**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repo
4. Set **Main file path**: `app.py`
5. Go to **Settings → Secrets** and add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   PINECONE_API_KEY = "..."
   PINECONE_INDEX_NAME = "rag-assistant"
   PINECONE_CLOUD = "aws"
   PINECONE_REGION = "us-east-1"
   ```
6. Click **Deploy**

### Render

1. Push your code to **GitHub**
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your repo
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
5. Add environment variables in **Environment** tab
6. Deploy

### Railway

1. Push your code to **GitHub**
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub Repo**
3. Add environment variables
4. Add a **Dockerfile** or set:
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. Deploy

---

## 🧩 How It All Works

### 1. How Embeddings Work

Embeddings convert text into **numerical vectors** (lists of numbers) that capture semantic meaning.

```
"Machine learning is great" → [0.012, -0.034, 0.056, ..., 0.078]  (1536 numbers)
"AI is awesome"             → [0.011, -0.033, 0.055, ..., 0.079]  (very similar!)
"I like pizza"              → [0.089, 0.042, -0.071, ..., -0.012] (very different)
```

- **Similar meanings → similar vectors** (close in 1536-dimensional space)
- We use OpenAI's `text-embedding-3-small` model (1536 dimensions)
- This is what makes semantic search possible — you search by meaning, not keywords

### 2. How Pinecone Stores Vectors

Pinecone is a **managed vector database** optimized for similarity search:

- **Index**: A collection of vectors (like a database table)
- **Vectors**: Each chunk gets stored as a vector + metadata (source file, page number)
- **Namespaces**: Logical partitions within an index (for multi-tenant apps)
- **ANN Search**: Uses Approximate Nearest Neighbor algorithms for fast retrieval

```
Pinecone Index: "rag-assistant"
├── Vector 1: [0.01, -0.03, ...] + {source: "report.pdf", page: 1}
├── Vector 2: [0.02, -0.01, ...] + {source: "report.pdf", page: 2}
└── Vector 3: [0.05,  0.04, ...] + {source: "paper.pdf",  page: 1}
```

### 3. How the RAG Pipeline Works

RAG = **Retrieval-Augmented Generation**. Instead of relying solely on the LLM's training data, we:

1. **Retrieve** relevant documents from a knowledge base (Pinecone)
2. **Augment** the LLM's prompt with this context
3. **Generate** an answer grounded in the retrieved documents

This eliminates hallucination and lets the AI answer about YOUR specific documents.

### 4. How Retrieval Works

1. User's question is embedded into a vector
2. Pinecone computes cosine similarity between the query vector and all stored vectors
3. Top-k most similar chunks are returned
4. These chunks become the "context" for the LLM

**Cosine Similarity** measures the angle between two vectors:
- `1.0` = identical direction (perfect match)
- `0.0` = perpendicular (unrelated)
- `-1.0` = opposite (contradictory)

### 5. How LangChain Connects Everything

LangChain is the **orchestration framework** that glues all components together:

| Component | LangChain Abstraction | Our Implementation |
|---|---|---|
| LLM | `ChatOpenAI` | GPT-4.1-mini |
| Embeddings | `OpenAIEmbeddings` | text-embedding-3-small |
| Vector Store | `PineconeVectorStore` | Pinecone serverless |
| Text Splitter | `RecursiveCharacterTextSplitter` | 1000 chars, 200 overlap |
| Memory | `ConversationBufferMemory` | Session-scoped |
| Documents | `Document` | PDF pages with metadata |

---

## 📝 License

MIT License — feel free to use this in your portfolio, resume, or production apps.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request
