# AI Research & Code Copilot

A production-grade multi-agent AI Research & Code Copilot system powered by **Endee Vector Database**. This system provides advanced document ingestion, vector search, RAG (Retrieval-Augmented Generation), and a multi-agent AI system for research assistance.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green)
![React](https://img.shields.io/badge/React-18.2-blue)
![Endee](https://img.shields.io/badge/Endee-VectorDB-purple)

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [System Flow](#system-flow)
- [Features](#features)
- [Endee Usage](#endee-usage)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Example Queries](#example-queries)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

---

## Problem Statement

Traditional document search systems rely on keyword matching, which fails to capture semantic meaning. This project addresses the need for:

1. **Semantic Search**: Find relevant documents based on meaning, not just keywords
2. **Intelligent Q&A**: Answer questions using context from uploaded documents
3. **Multi-Agent Assistance**: Specialized AI agents for retrieval, analysis, and recommendations
4. **Production-Ready**: Scalable architecture suitable for internship evaluation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Frontend (React + Tailwind)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐                   │
│  │ Upload Page │  │ Query Page  │  │ Recommendations  │                   │
│  └─────────────┘  └─────────────┘  └──────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Backend                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          API Routes                                   │  │
│  │  /documents/*  /search  /query  /agents/*  /index/*                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Ingestion  │  │ Embedding   │  │    RAG     │  │   Agents    │      │
│  │  Service   │  │  Service    │  │  Pipeline   │  │             │      │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                      │                                      │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Endee Vector Database (Docker)                           │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  HNSW Index │ Metadata Filters │ Similarity Search │ Quantization │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

1. **Frontend (React + Tailwind)**
   - Clean, modern UI
   - Upload documents (PDF, TXT, GitHub)
   - Query with RAG
   - View recommendations

2. **FastAPI Backend**
   - RESTful API
   - Async operations
   - Error handling
   - Request validation

3. **Endee Vector Database**
   - High-performance vector storage
   - HNSW indexing
   - Metadata filtering
   - Multiple distance metrics

4. **AI Services**
   - **Ingestion**: PDF/TXT/GitHub processing
   - **Embedding**: sentence-transformers
   - **RAG**: Context-aware问答
   - **Agents**: Retrieval, Analysis, Recommendation

---

## System Flow

### Document Upload Flow

```
User Upload → File Processing → Smart Chunking → Embedding Generation → Endee Storage
     │              │                 │                │                    │
     ▼              ▼                 ▼                ▼                    ▼
  PDF/TXT       Extract Text      Split into      Generate          Store vectors
  /GitHub        Content          Overlapping     384-dim           with metadata
                                  Chunks          Embeddings        in Endee
```

### Query Flow

```
User Query → Embed Query → Vector Search → Context Retrieval → LLM Generation → Response
    │            │            │               │                   │            │
    ▼            ▼            ▼               ▼                   ▼            ▼
  Natural     Convert to    Find similar   Fetch top K         Generate    Formatted
  Language    384-dim       vectors from   chunks from         answer with  response
  Question    Embedding     Endee          Endee               context     with sources
```

### Agent Pipeline Flow

```
Query → Retrieval Agent → Analysis Agent → Recommendation Agent → Combined Response
         │                   │                   │                       │
         ▼                   ▼                   ▼                       ▼
    Find relevant      Extract themes      Generate            Complete
    documents          & insights          suggestions         AI response
```

---

## Features

### ✅ Document Ingestion
- PDF file processing (via PyPDF2)
- Text file support (.txt, .md, .json)
- GitHub repository file fetching
- Smart chunking with configurable overlap

### ✅ Embedding Generation
- sentence-transformers (all-MiniLM-L6-v2)
- Batch processing for efficiency
- CPU/CUDA support

### ✅ Endee Vector Database
- **Core component** - not optional
- HNSW indexing for fast search
- Metadata filtering (numeric, categorical)
- Multiple space types (cosine, L2, IP)
- Quantization support (int8, int16, float32)

### ✅ RAG Pipeline
- Top-K document retrieval
- Context formatting with truncation
- LLM integration (OpenAI/Anthropic)
- Source citation in responses

### ✅ AI Agents
1. **Retrieval Agent**: Find relevant documents
2. **Analysis Agent**: Extract insights & themes
3. **Recommendation Agent**: Generate actionable suggestions

### ✅ Production Features
- Environment-based configuration
- Comprehensive logging
- Error handling
- Modular architecture

---

## Endee Usage

Endee is the **core vector database** for this system. Here's how it's integrated:

### Index Creation

```
python
# Create index in Endee
endee_client.create_index(
    index_name="research_copilot",
    dimension=384,  # Matches embedding dimension
    space_type="cosine",
    m=16,  # HNSW M parameter
    ef_construct=200,
    precision="int16"
)
```

### Vector Insertion

```
python
# Insert document embeddings
vectors = [
    {
        "id": "chunk_1",
        "vector": [0.1, 0.2, ...],  # 384-dim embedding
        "metadata": {
            "doc_id": "doc_1",
            "title": "My Document",
            "source": "file.pdf"
        }
    }
]
endee_client.insert_vectors("research_copilot", vectors)
```

### Similarity Search

```
python
# Search for similar documents
results = endee_client.search(
    index_name="research_copilot",
    query_vector=query_embedding,
    k=5,
    filter_expr={"source": {"$in": ["file1.pdf", "file2.pdf"]}}
)
```

### Metadata Filtering

Endee supports powerful filtering:

```
python
# Numeric filters
filter_expr = {"page_count": {"$gte": 10, "$lte": 50}}

# Category filters
filter_expr = {"file_type": {"$in": ["pdf", "txt"]}}

# Boolean filters
filter_expr = {"is_indexed": True}
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- OpenAI or Anthropic API key (for LLM)

### Step 1: Clone and Setup

```
bash
# Navigate to project directory
cd endee-intern-project

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### Step 2: Start Endee Vector Database

```
bash
# Start Endee using Docker
docker run -d \
  --name endee-server \
  -p 8080:8080 \
  -v endee-data:/data \
  endeeio/endee-server:latest

# Verify it's running
curl http://localhost:8080/api/v1/health
```

### Step 3: Install Backend Dependencies

```
bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Start Backend

```
bash
# Start FastAPI server
uvicorn backend.main:app --reload --port 8000

# Access API docs at http://localhost:8000/docs
```

### Step 5: Start Frontend

```
bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Access at http://localhost:3000
```

---

## API Endpoints

### Health & Status
```
bash
GET /api/v1/health     # Check system health
GET /api/v1/status      # Get system status
```

### Index Management
```
bash
POST   /api/v1/index/create  # Create index
GET    /api/v1/index/list    # List indexes
DELETE /api/v1/index/{name}  # Delete index
```

### Document Ingestion
```
bash
POST /api/v1/documents/upload   # Upload file (PDF, TXT, etc.)
POST /api/v1/documents/text     # Upload raw text
POST /api/v1/documents/github   # Upload GitHub repo
```

### Search & Query
```
bash
POST /api/v1/search   # Vector similarity search
POST /api/v1/query    # RAG-powered Q&A
```

### AI Agents
```
bash
POST /api/v1/agents/retrieval      # Retrieval Agent
POST /api/v1/agents/analysis        # Analysis Agent
POST /api/v1/agents/recommendation # Recommendation Agent
POST /api/v1/agents/pipeline       # Full Agent Pipeline
```

### Embeddings
```
bash
POST /api/v1/embeddings  # Generate embeddings
```

---

## Example API Calls

### Upload a Document

```
bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf" \
  -F "title=My Research Paper"
```

### Query with RAG

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "top_k": 5
  }'
```

### Use Agent Pipeline

```
bash
curl -X POST "http://localhost:8000/api/v1/agents/pipeline" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find information about machine learning",
    "top_k": 5
  }'
```

---

## Testing

### Automated Tests

Run the automated API tests using the test script:

```
bash
python test_api.py
```

### Manual Test Cases

You can also test the API endpoints manually using the following commands:

#### 1. Health Check
```
bash
curl -X GET http://localhost:8000/api/v1/health
```

#### 2. Get Status
```
bash
curl -X GET http://localhost:8000/api/v1/status
```

#### 3. Upload Text Document
```
bash
curl -X POST http://localhost:8000/api/v1/documents/text \
  -H "Content-Type: application/json" \
  -d '{"title":"Test Doc","content":"AI and ML","source_type":"txt"}'
```

#### 4. Search Documents
```
bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence", "top_k": 5}'
```

#### 5. Generate Embeddings
```
bash
curl -X POST http://localhost:8000/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '["Hello world", "Machine learning"]'
```

#### 6. Retrieval Agent
```
bash
curl -X POST http://localhost:8000/api/v1/agents/retrieval \
  -H "Content-Type: application/json" \
  -d '{"query": "AI", "agent_type": "retrieval", "top_k": 3}'
```

#### 7. Recommendation Agent
```
bash
curl -X POST http://localhost:8000/api/v1/agents/recommendation \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning"}'
```

### Test Results Summary

| Test | Endpoint | Status |
|------|----------|--------|
| Health Check | GET /api/v1/health | ✅ PASS |
| Get Status | GET /api/v1/status | ✅ PASS |
| Upload Text | POST /api/v1/documents/text | ⚠️ Requires Endee |
| Search | POST /api/v1/search | ⚠️ Requires Endee |
| Embeddings | POST /api/v1/embeddings | ✅ PASS |
| Retrieval Agent | POST /api/v1/agents/retrieval | ✅ PASS |
| Recommendation Agent | POST /api/v1/agents/recommendation | ✅ PASS |

### Browser Testing

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

### Prerequisites for Full Testing

1. **Start Docker Desktop** - Required for Endee
2. **Start Endee**: `docker-compose up -d`
3. **Verify Endee**: `curl http://localhost:8080/api/v1/health`

---

## Troubleshooting

### Common Issues

1. **Endee Connection Failed**
   - Ensure Docker Desktop is running
   - Run: `docker-compose up -d`

2. **Embedding Model Error**
   - Install: `pip install transformers==4.36.0`

---

## Example Queries

Once documents are uploaded, try these example queries:

1. **General Search**
   - "What documents do we have about neural networks?"
   - "Find information about API design patterns"

2. **Specific Questions**
   - "What is the main conclusion of the research paper?"
   - "How does the code implement authentication?"

3. **Code Assistance**
   - "Show me examples of error handling in Python"
   - "What are the best practices for REST APIs?"

4. **Recommendations**
   - "What should I learn next about machine learning?"
   - "Give me suggestions for improving my code"

---

## Configuration

### Environment Variables (.env)

```
env
# Endee Vector Database
ENDEE_BASE_URL=http://localhost:8080
ENDEE_AUTH_TOKEN=
ENDEE_INDEX_NAME=research_copilot
ENDEE_DIMENSION=384

# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4-turbo-preview

# Or Anthropic
# ANTHROPIC_API_KEY=your_key_here

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `ENDEE_BASE_URL` | http://localhost:8080 | Endee API URL |
| `ENDEE_DIMENSION` | 384 | Embedding dimension |
| `CHUNK_SIZE` | 1000 | Text chunk size |
| `CHUNK_OVERLAP` | 200 | Chunk overlap |
| `TOP_K_DOCUMENTS` | 5 | Default search results |
| `LLM_PROVIDER` | openai | LLM provider |

---

## Project Structure

```
endee-intern-project/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # FastAPI routes
│   ├── agents/
│   │   ├── __init__.py
│   │   └── agents.py           # AI Agents
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration
│   │   └── logging.py          # Logging setup
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedding_service.py # Embeddings
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── document_processor.py # Document processing
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   ├── rag/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py     # RAG pipeline
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── endee_client.py     # Endee client
│   ├── __init__.py
│   └── main.py                 # FastAPI app
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── Navbar.jsx
│   │   ├── pages/
│   │   │   ├── QueryPage.jsx
│   │   │   ├── RecommendationsPage.jsx
│   │   │   └── UploadPage.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
├── .env.example
├── requirements.txt
└── README.md
```

---

## Docker Compose (Optional)

Create a `docker-compose.yml` for the full stack:

```yaml
version: '3.8'

services:
  endee:
    image: endeeio/endee-server:latest
    ports:
      - "8080:8080"
    volumes:
      - endee-data:/data

  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENDEE_BASE_URL=http://endee:8080
    depends_on:
      - endee

volumes:
  endee-data:
```

---

## License

Apache License 2.0

---

## Acknowledgments

- [Endee Vector Database](https://github.com/endee-io/endee) - High-performance vector search
- [sentence-transformers](https://sbert.net/) - State-of-the-art embeddings
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [LangChain](https://langchain.com/) - LLM application framework
