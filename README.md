# PaperMind

Multi-agent autonomous research system for analyzing arXiv papers using MCP-based coordination, RAG architecture, and specialized research agents.

## Overview

PaperMind ingests arXiv papers, processes them into a vector database, and employs five specialized agents to produce comprehensive research reports including literature reviews, method comparisons, results analysis, critical assessments, and research gap synthesis.

## Architecture

The system consists of:

- **Ingestion Pipeline**: arXiv API integration, PDF parsing, text chunking, and embedding generation
- **Vector Store**: FAISS-based similarity search over paper chunks
- **MCP Controller**: Multi-agent orchestration system coordinating specialized research agents
- **Research Agents**: Five domain-specific agents (Literature, Methods, Results, Critique, Synthesis)
- **API Server**: FastAPI backend exposing ingestion and research endpoints
- **UI**: Streamlit interface for interactive research

## Installation

```bash
git clone https://github.com/yourusername/PaperMind.git
cd PaperMind
pip install -r requirements.txt
```

## Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=your_key_here
```

Alternatively, the system will fall back to SentenceTransformers for local embedding generation if no API key is provided.

## Usage

### Command Line

**Ingest papers from arXiv:**

```bash
python -m papermind.main ingest "transformer neural networks" --max-results 10 --category cs.LG
```

**Conduct research:**

```bash
python -m papermind.main research "What are the latest advances in transformer architectures?"
```

### API Server

Start the FastAPI server:

```bash
uvicorn papermind.api.server:app --reload
```

Access API documentation at `http://localhost:8000/docs`

**Ingest papers:**

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanisms", "max_results": 10}'
```

**Generate research report:**

```bash
curl -X POST "http://localhost:8000/research" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do different attention mechanisms compare?"}'
```

### Web UI

Start the Streamlit interface:

```bash
streamlit run papermind/ui/app.py
```

Access the UI at `http://localhost:8501`

## Research Agents

1. **LiteratureAgent**: Analyzes related work, identifies key papers, and summarizes contributions
2. **MethodsAgent**: Compares methodologies, techniques, and experimental approaches
3. **ResultsAgent**: Analyzes experimental results, benchmarks, and performance metrics
4. **CritiqueAgent**: Identifies limitations, weaknesses, and unresolved challenges
5. **SynthesisAgent**: Integrates findings and identifies research gaps and opportunities

## Project Structure

```
papermind/
├── data/                   # Downloaded papers and processed text
├── embeddings/             # FAISS index and metadata
├── agents/                 # Specialized research agents
├── ingestion/              # Data pipeline modules
├── vectorstore/            # Vector database implementation
├── mcp/                    # Multi-agent controller
├── api/                    # FastAPI server
└── ui/                     # Streamlit interface
```

## Dependencies

- Python 3.10+
- FastAPI, Uvicorn
- Streamlit
- OpenAI API or SentenceTransformers
- FAISS
- PyPDF

See `requirements.txt` for complete list.

## Development

The system is modular and extensible. To add new agents:

1. Create agent class in `papermind/agents/`
2. Implement `execute(query, context, client, model)` method
3. Register agent with MCP controller

## License

MIT License

## Citation

If you use PaperMind in your research, please cite:

