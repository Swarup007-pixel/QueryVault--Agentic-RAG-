# QueryVault--Agentic-RAG-
A modular AI assistant built with agentic RAG architecture, enabling intelligent document retrieval, reasoning, and multi-step question answering.

## Features

- Agentic Retrieval Workflow
- Document Processing and Chunking
- Semantic Search
- Context-Aware Question Answering
- Multi-Step Reasoning
- Source-Aware Response Generation
- FastAPI Backend
- Streamlit Frontend
- Local LLM Inference using Gemma 4 E4B

## Tech Stack

### LLM
- Gemma 4 E4B

### Model Runtime
- Ollama

### Framework
- LangChain

### Vector Database
- ChromaDB

### Embeddings
- Sentence Transformers

### Backend
- FastAPI

### Frontend
- Streamlit

## Architecture

```text
Documents
    │
    ▼
Document Processing
    │
    ▼
Chunking
    │
    ▼
Embeddings
    │
    ▼
ChromaDB Vector Store
    │
    ▼
Agentic Retrieval Layer
    │
    ▼
Gemma 4 E4B
    │
    ▼
Grounded Response
```

## Installation

### Install Required Dependencies

```bash
pip install langchain
pip install langchain-ollama
pip install chromadb
pip install sentence-transformers
pip install fastapi
pip install uvicorn
pip install streamlit
```

Or:

```bash
pip install -r requirements.txt
```

### Install Ollama

Install Ollama on your system.

### Download Gemma 4 E4B

```bash
ollama pull gemma4:e4b
```

## Project Structure

```text
QueryVault-Agentic-RAG/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── model.py                 # Initializes and manages the local Gemma4:e4b LLM
│   ├── document_processor.py    # Loads PDFs, splits text, and generates embeddings
│   ├── vector_store.py          # Creates, saves, loads, and queries the FAISS vector store
│   ├── rag_pipeline.py          # Orchestrates the complete RAG workflow from retrieval to response generation
│   ├── agent.py                 # Handles user queries and communicates with the RAG pipeline
│   └── main.py                  # Entry point of the application
│
├── data/                        # Stores uploaded PDFs and generated vector databases
│
└── docs/                        # Project documentation and architecture diagrams
```


## Current Status

Active Development

## Planned Features

- Agentic Query Planning
- Multi-Step Retrieval
- Multi-Document Reasoning
- Citation-Based Responses
- Memory-Augmented Conversations
- Evaluation and Monitoring Dashboard

## Author

Swarup Mohapatra