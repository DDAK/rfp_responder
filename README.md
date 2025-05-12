# Intelligent RFP Agent System

Notes:
An intelligent system for automating the RFP response process using AI and knowledge graphs.

A scalable job queue system using PostgreSQL and FastAPI where:
- Each job triggers an AI agent to generate a document.
- PostgreSQL's NOTIFY/LISTEN is used for reactive job detection.
- Each job spawns a Dockerized worker container (up to a max limit).
- Workers terminate automatically after job completion if the queue is empty.


## Features

- Automated RFP document processing (PDF, DOCX, HTML, XLSX, ZIP)
- Requirement extraction and analysis
- Knowledge base management with Graph RAG
- Automated proposal generation
- Compliance checking and validation
- Proposal improvement based on feedback

## Architecture

The system is built using:
- LangGraph for workflow orchestration
- LangChain for LLM integration
- Neo4j for knowledge graph storage
- LightRAG for graph-based retrieval
- FastAPI for the REST API
- Docker for containerization
- Ollama to serve the models
- Also uses Groq (conditionally)


## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <folder>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the services using Docker Compose:
```bash
docker-compose up -d
```