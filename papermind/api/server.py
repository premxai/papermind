from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os

from papermind.ingestion.arxiv_loader import ArxivLoader
from papermind.ingestion.pdf_parser import PDFParser
from papermind.ingestion.chunker import Chunker
from papermind.ingestion.embedder import Embedder
from papermind.vectorstore.faiss_store import FAISSStore
from papermind.mcp.controller import MCPController
from papermind.agents.literature import LiteratureAgent
from papermind.agents.methods import MethodsAgent
from papermind.agents.results import ResultsAgent
from papermind.agents.critique import CritiqueAgent
from papermind.agents.synthesis import SynthesisAgent


app = FastAPI(title="PaperMind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None
embedder = None
controller = None


class IngestRequest(BaseModel):
    query: str
    max_results: int = 10
    category: Optional[str] = None
    year: Optional[int] = None


class ResearchRequest(BaseModel):
    query: str
    retrieval_k: int = 15


class IngestResponse(BaseModel):
    status: str
    papers_found: int
    papers_processed: int
    chunks_created: int


class ResearchResponse(BaseModel):
    query: str
    literature_review: Dict
    methods_analysis: Dict
    results_analysis: Dict
    critique: Dict
    synthesis: Dict
    sources: List[Dict]
    num_sources: int


@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup."""
    global vector_store, embedder, controller
    
    embedder = Embedder()
    dimension = 1536 if embedder.use_openai else 384
    vector_store = FAISSStore(dimension=dimension)
    
    controller = MCPController(
        vector_store=vector_store,
        embedder=embedder
    )
    
    controller.register_agent('literature', LiteratureAgent())
    controller.register_agent('methods', MethodsAgent())
    controller.register_agent('results', ResultsAgent())
    controller.register_agent('critique', CritiqueAgent())
    controller.register_agent('synthesis', SynthesisAgent())


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "PaperMind API", "version": "1.0.0"}


@app.get("/status")
async def status():
    """Get system status."""
    stats = vector_store.get_stats()
    return {
        "status": "online",
        "vector_store": stats
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_papers(request: IngestRequest):
    """
    Ingest papers from arXiv into the vector store.
    
    Args:
        request: Ingest request with query and filters
        
    Returns:
        Ingestion status
    """
    try:
        loader = ArxivLoader()
        papers = loader.search(
            query=request.query,
            max_results=request.max_results,
            category=request.category,
            year=request.year
        )
        
        papers = loader.download_papers(papers)
        
        parser = PDFParser()
        papers = parser.parse_papers(papers)
        
        chunker = Chunker()
        chunks = chunker.chunk_papers(papers)
        
        chunks = embedder.embed_chunks(chunks)
        
        vector_store.add_chunks(chunks)
        vector_store.save()
        
        return IngestResponse(
            status="success",
            papers_found=len(papers),
            papers_processed=len([p for p in papers if 'text' in p]),
            chunks_created=len(chunks)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research", response_model=ResearchResponse)
async def conduct_research(request: ResearchRequest):
    """
    Conduct autonomous research using multi-agent system.
    
    Args:
        request: Research request with query
        
    Returns:
        Comprehensive research report
    """
    try:
        if vector_store.index.ntotal == 0:
            raise HTTPException(
                status_code=400,
                detail="No papers in vector store. Please ingest papers first."
            )
        
        report = controller.orchestrate(
            query=request.query,
            retrieval_k=request.retrieval_k
        )
        
        return ResearchResponse(**report)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory")
async def get_memory():
    """Get controller memory and conversation history."""
    memory = controller.get_memory()
    return {
        "messages": memory.get_messages(),
        "agent_results": memory.get_all_agent_results(),
        "queries": memory.query_history
    }


@app.delete("/memory")
async def clear_memory():
    """Clear controller memory."""
    controller.clear_memory()
    return {"status": "memory cleared"}


@app.delete("/vectorstore")
async def clear_vectorstore():
    """Clear vector store."""
    vector_store.clear()
    vector_store.save()
    return {"status": "vector store cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
