# 🚀 RAG + Agentic Pipeline

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline built with **FAISS**, **Sentence Transformers**, **FastAPI**, and **LangGraph**.

The system combines semantic search with a multi-agent workflow to generate grounded, self-verified responses instead of relying on a single LLM call.

---

## ✨ Features

- 📄 Multi-format document ingestion (`.pdf`, `.txt`, `.md`)
- ✂️ Recursive text chunking with overlap
- 🔍 Semantic search using FAISS
- 🤖 LangGraph Agentic workflow
  - Researcher
  - Writer
  - Reviewer
- 🔄 Automatic revision loop
- ⚡ FastAPI REST API
- 🐳 Docker support
- 🧪 Unit tests
- 🔐 API key authentication
- 📦 Modular architecture

---

# 🏗️ Architecture

```text
                         ┌──────────────────────────────────────────────┐
                         │                Ingestion                     │
 Documents (.pdf/.txt/.md)
            │
            ▼
      loader.py
            │
            ▼
      chunker.py
            │
            ▼
     embedder.py
            │
            ▼
     Sentence Transformers
            │
            ▼
     ┌──────────────────────┐
     │     FAISS Vector DB  │
     │ Cosine Similarity    │
     │ Persistent Storage   │
     └──────────┬───────────┘
                ▲
                │
         Query Embedding
                │
                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      LangGraph Agent Workflow                              │
│                                                                            │
│  ┌─────────────┐      ┌────────────┐      ┌─────────────┐                  │
│  │ Researcher  │ ───► │   Writer   │ ───► │  Reviewer   │                  │
│  │ (RAG Tool)  │      │ Drafts     │      │ Fact Check  │                  │
│  └─────────────┘      └─────▲──────┘      └─────┬───────┘                  │
│                              │                  │                           │
│                              └──── Revise ◄─────┘                           │
│                                 (Maximum 2 loops)                          │
│                                                                            │
│                              Final Approved Answer                         │
└────────────────────────────────────────────────────────────────────────────┘
                │
                ▼
        FastAPI Endpoints

        POST /query
        POST /agent/query
```

---

# 💡 Why Agentic RAG?

Traditional RAG performs:

```
User Query
     │
     ▼
 Retrieve
     │
     ▼
    LLM
     │
     ▼
  Response
```

Agentic RAG introduces reasoning and verification:

```
User Query
     │
     ▼
Researcher
     │
     ▼
 Writer
     │
     ▼
Reviewer
     │
     ├── Approved ─────────► Final Answer
     │
     └── Needs Revision
              │
              ▼
           Writer
```

### Benefits

- Grounded answers
- Reduced hallucinations
- Self-verification
- Better factual accuracy
- Modular architecture
- Easy to extend

---

# 📂 Project Structure

```text
app/
│
├── config.py                     # Configuration
│
├── ingestion/
│   ├── loader.py                 # Load PDF/TXT/Markdown
│   └── chunker.py                # Recursive text splitter
│
├── embeddings/
│   └── embedder.py               # SentenceTransformer wrapper
│
├── vectorstore/
│   ├── faiss_store.py            # FAISS implementation
│   └── pgvector_store.py         # PostgreSQL pgvector implementation
│
├── rag/
│   └── pipeline.py               # Complete RAG pipeline
│
├── agents/
│   ├── state.py                  # LangGraph state
│   ├── tools.py                  # RAG + LLM tools
│   ├── nodes.py                  # Researcher / Writer / Reviewer
│   └── graph.py                  # Agent workflow
│
├── api/
│   └── main.py                   # FastAPI application
│
cli.py                            # Command line interface

tests/
├── test_chunker.py
└── test_faiss_store.py

Dockerfile
docker-compose.yml
requirements.txt
README.md
```

---

# 🛠️ Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| API Framework | FastAPI |
| Agent Framework | LangGraph |
| LLM | OpenAI |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| Optional Vector DB | PGVector |
| Testing | PyTest |
| Containerization | Docker |

---

# 🚀 Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/somrajmondal/Rag-Agentic-Pipeline.git

cd Rag-Agentic-Pipeline
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure the Project

Create a local configuration file from the provided template.

### Windows

```powershell
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cp .env.example .env
```

Update the values in `.env` before running the application.

> **Important:** The `.env` file is excluded from version control and should never be committed.

---

## 5. Ingest Documents

```bash
python cli.py ingest ./data/documents
```

Supported document formats:

- PDF
- Markdown
- Text

---

## 6. Query the Knowledge Base

### Standard RAG

```bash
python cli.py query "What are the payment terms?"
```

Pipeline

```
Query
   ↓
Retrieve
   ↓
LLM
   ↓
Answer
```

---

### Agentic RAG

```bash
python cli.py agent "What are the payment terms?"
```

Pipeline

```
Query
   ↓
Researcher
   ↓
Writer
   ↓
Reviewer
   ↓
Revision (if required)
   ↓
Final Response
```

---

## 7. Run the API

```bash
uvicorn app.api.main:app --reload
```

Open

```
http://localhost:8000/docs
```

for Swagger documentation.

---

# 📡 API Endpoints

| Endpoint | Description |
|-----------|-------------|
| POST `/ingest/directory` | Index documents |
| POST `/query` | Standard RAG query |
| POST `/agent/query` | Agentic RAG query |

---

## Example Request

### Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest/directory \
-H "Content-Type: application/json" \
-H "X-API-Key: <your_api_key>" \
-d "{\"directory\":\"./data/documents\"}"
```

---

### Standard Query

```bash
curl -X POST http://localhost:8000/query \
-H "Content-Type: application/json" \
-H "X-API-Key: <your_api_key>" \
-d "{\"query\":\"Explain the payment terms.\"}"
```

---

### Agentic Query

```bash
curl -X POST http://localhost:8000/agent/query \
-H "Content-Type: application/json" \
-H "X-API-Key: <your_api_key>" \
-d "{\"query\":\"Explain the payment terms.\"}"
```

---

# 🤖 Agent Workflow

## Researcher

Responsibilities

- Retrieves relevant documents
- Searches the vector database
- Returns contextual evidence

---

## Writer

Responsibilities

- Generates grounded answers
- Uses retrieved context only
- Includes citations where applicable

---

## Reviewer

Responsibilities

- Validates generated response
- Detects hallucinations
- Requests revisions when necessary

---

### Revision Flow

```text
Research
     │
     ▼
 Write
     │
     ▼
Review
     │
 ┌───┴──────────┐
 │              │
Approved      Revise
 │              │
 ▼              │
Done ◄──────────┘
```

Maximum revisions:

```
2
```

---

# 🔄 Swappable Components

## Embedding Model

Replace the embedding implementation inside:

```
app/embeddings/embedder.py
```

---

## Vector Database

Current

- FAISS

Optional

- PGVector

Implement the following interface:

- `init_schema()`
- `upsert()`
- `similarity_search()`
- `count()`

---

## Add More Agents

Create a new node inside

```
app/agents/
```

Then connect it inside

```
graph.py
```

The graph is fully modular and easy to extend.

---

# 🧪 Running Tests

Run all tests

```bash
pytest
```

Run a single test

```bash
pytest tests/test_chunker.py
```

```bash
pytest tests/test_faiss_store.py
```

---

# 🐳 Docker

Build

```bash
docker compose build
```

Start

```bash
docker compose up
```

Detached mode

```bash
docker compose up -d
```

Stop

```bash
docker compose down
```

---

# 📈 Roadmap

- Hybrid Search (BM25 + Vector Search)
- Reranking
- Streaming Responses
- Multi-LLM Support
- Redis Cache
- Conversation Memory
- Citation Confidence Scores
- Kubernetes Deployment
- Monitoring & Tracing
- Tool Calling
- MCP Integration

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push your branch.
5. Open a Pull Request.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Somraj Mondal**

**AI/ML Engineer | Generative AI | RAG | LangGraph | FastAPI | Vector Databases**

If you found this project helpful, consider giving it a ⭐ on GitHub.