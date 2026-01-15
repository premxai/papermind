import argparse
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


def ingest_papers(query: str, max_results: int = 10, category: str = None, year: int = None):
    """Ingest papers from arXiv into vector store."""
    print(f"Searching arXiv for: {query}")
    
    loader = ArxivLoader()
    papers = loader.search(query, max_results, category, year)
    print(f"Found {len(papers)} papers")
    
    print("Downloading PDFs...")
    papers = loader.download_papers(papers)
    
    print("Parsing PDFs...")
    parser = PDFParser()
    papers = parser.parse_papers(papers)
    print(f"Successfully parsed {len(papers)} papers")
    
    print("Chunking papers...")
    chunker = Chunker()
    chunks = chunker.chunk_papers(papers)
    print(f"Created {len(chunks)} chunks")
    
    print("Generating embeddings...")
    embedder = Embedder()
    chunks = embedder.embed_chunks(chunks)
    
    print("Storing in vector database...")
    dimension = 1536 if embedder.use_openai else 384
    vector_store = FAISSStore(dimension=dimension)
    vector_store.add_chunks(chunks)
    vector_store.save()
    
    print(f"Ingestion complete! {len(chunks)} chunks stored.")
    return len(chunks)


def conduct_research(query: str, retrieval_k: int = 15):
    """Conduct research using multi-agent system."""
    print(f"Research query: {query}\n")
    
    embedder = Embedder()
    dimension = 1536 if embedder.use_openai else 384
    vector_store = FAISSStore(dimension=dimension)
    
    if vector_store.index.ntotal == 0:
        print("Error: No papers in vector store. Please ingest papers first.")
        return
    
    print(f"Vector store contains {vector_store.index.ntotal} chunks\n")
    
    controller = MCPController(
        vector_store=vector_store,
        embedder=embedder
    )
    
    controller.register_agent('literature', LiteratureAgent())
    controller.register_agent('methods', MethodsAgent())
    controller.register_agent('results', ResultsAgent())
    controller.register_agent('critique', CritiqueAgent())
    controller.register_agent('synthesis', SynthesisAgent())
    
    print("Running multi-agent analysis...\n")
    report = controller.orchestrate(query, retrieval_k)
    
    print("=" * 80)
    print("RESEARCH REPORT")
    print("=" * 80)
    
    print(f"\nQuery: {report['query']}")
    print(f"Sources: {report['num_sources']} papers\n")
    
    print("-" * 80)
    print("LITERATURE REVIEW")
    print("-" * 80)
    print(report['literature_review'].get('analysis', ''))
    
    print("\n" + "-" * 80)
    print("METHODS ANALYSIS")
    print("-" * 80)
    print(report['methods_analysis'].get('analysis', ''))
    
    print("\n" + "-" * 80)
    print("RESULTS ANALYSIS")
    print("-" * 80)
    print(report['results_analysis'].get('analysis', ''))
    
    print("\n" + "-" * 80)
    print("CRITICAL ANALYSIS")
    print("-" * 80)
    print(report['critique'].get('analysis', ''))
    
    print("\n" + "-" * 80)
    print("SYNTHESIS")
    print("-" * 80)
    print(report['synthesis'].get('analysis', ''))
    
    print("\n" + "-" * 80)
    print("SOURCES")
    print("-" * 80)
    for source in report['sources']:
        print(f"\n{source['title']}")
        print(f"Authors: {', '.join(source['authors'][:3])}...")
        print(f"Published: {source['published']}")
        print(f"ID: {source['paper_id']}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="PaperMind: Autonomous Research System")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    ingest_parser = subparsers.add_parser('ingest', help='Ingest papers from arXiv')
    ingest_parser.add_argument('query', type=str, help='Search query')
    ingest_parser.add_argument('--max-results', type=int, default=10, help='Maximum results')
    ingest_parser.add_argument('--category', type=str, help='arXiv category filter')
    ingest_parser.add_argument('--year', type=int, help='Year filter')
    
    research_parser = subparsers.add_parser('research', help='Conduct research')
    research_parser.add_argument('query', type=str, help='Research question')
    research_parser.add_argument('--retrieval-k', type=int, default=15, help='Number of chunks to retrieve')
    
    args = parser.parse_args()
    
    if args.command == 'ingest':
        ingest_papers(args.query, args.max_results, args.category, args.year)
    elif args.command == 'research':
        conduct_research(args.query, args.retrieval_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
