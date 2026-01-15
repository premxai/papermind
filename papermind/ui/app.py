import streamlit as st
import requests
import json
from typing import Dict


API_URL = "http://localhost:8000"


st.set_page_config(
    page_title="PaperMind",
    page_icon="📚",
    layout="wide"
)


def check_api_status():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/status")
        return response.status_code == 200
    except:
        return False


def ingest_papers(query: str, max_results: int, category: str = None, year: int = None):
    """Ingest papers from arXiv."""
    payload = {
        "query": query,
        "max_results": max_results,
        "category": category if category else None,
        "year": year if year else None
    }
    
    response = requests.post(f"{API_URL}/ingest", json=payload)
    return response.json()


def conduct_research(query: str, retrieval_k: int = 15):
    """Conduct research using multi-agent system."""
    payload = {
        "query": query,
        "retrieval_k": retrieval_k
    }
    
    response = requests.post(f"{API_URL}/research", json=payload)
    return response.json()


def get_vector_store_status():
    """Get vector store statistics."""
    response = requests.get(f"{API_URL}/status")
    return response.json()


def clear_vector_store():
    """Clear vector store."""
    response = requests.delete(f"{API_URL}/vectorstore")
    return response.json()


def main():
    st.title("PaperMind")
    st.markdown("**Autonomous Research System for arXiv Papers**")
    
    if not check_api_status():
        st.error("API server is not running. Please start the server with: `uvicorn papermind.api.server:app`")
        return
    
    status = get_vector_store_status()
    
    with st.sidebar:
        st.header("System Status")
        st.metric("Papers Indexed", status.get("vector_store", {}).get("total_chunks", 0))
        st.metric("Vector Dimension", status.get("vector_store", {}).get("dimension", 0))
        
        if st.button("Clear Vector Store"):
            clear_vector_store()
            st.success("Vector store cleared")
            st.rerun()
        
        st.divider()
        
        st.header("Settings")
        retrieval_k = st.slider("Retrieval Context Size", 5, 30, 15)
    
    tab1, tab2 = st.tabs(["Ingest Papers", "Research"])
    
    with tab1:
        st.header("Ingest Papers from arXiv")
        
        with st.form("ingest_form"):
            ingest_query = st.text_input(
                "Search Query",
                placeholder="e.g., transformer neural networks"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_results = st.number_input("Max Results", min_value=1, max_value=50, value=10)
            
            with col2:
                category = st.text_input("Category (optional)", placeholder="e.g., cs.AI, cs.LG")
            
            with col3:
                year = st.number_input("Year (optional)", min_value=2000, max_value=2026, value=None, step=1)
            
            submit_ingest = st.form_submit_button("Ingest Papers")
        
        if submit_ingest:
            if not ingest_query:
                st.error("Please enter a search query")
            else:
                with st.spinner("Ingesting papers from arXiv..."):
                    try:
                        result = ingest_papers(
                            query=ingest_query,
                            max_results=max_results,
                            category=category if category else None,
                            year=year
                        )
                        
                        st.success(f"Ingestion completed!")
                        st.json(result)
                    except Exception as e:
                        st.error(f"Error during ingestion: {str(e)}")
    
    with tab2:
        st.header("Conduct Research")
        
        research_query = st.text_area(
            "Research Question",
            placeholder="e.g., What are the latest advances in transformer architectures for NLP?",
            height=100
        )
        
        if st.button("Generate Research Report", type="primary"):
            if not research_query:
                st.error("Please enter a research question")
            elif status.get("vector_store", {}).get("total_chunks", 0) == 0:
                st.warning("No papers in vector store. Please ingest papers first.")
            else:
                with st.spinner("Conducting multi-agent research analysis..."):
                    try:
                        report = conduct_research(research_query, retrieval_k=retrieval_k)
                        
                        st.success("Research report generated!")
                        
                        st.divider()
                        
                        st.subheader("Literature Review")
                        st.markdown(report["literature_review"].get("analysis", ""))
                        
                        st.divider()
                        
                        st.subheader("Methods Analysis")
                        st.markdown(report["methods_analysis"].get("analysis", ""))
                        
                        st.divider()
                        
                        st.subheader("Results Analysis")
                        st.markdown(report["results_analysis"].get("analysis", ""))
                        
                        st.divider()
                        
                        st.subheader("Critical Analysis")
                        st.markdown(report["critique"].get("analysis", ""))
                        
                        st.divider()
                        
                        st.subheader("Research Synthesis")
                        st.markdown(report["synthesis"].get("analysis", ""))
                        
                        st.divider()
                        
                        st.subheader("Sources")
                        st.write(f"**Total Sources:** {report['num_sources']}")
                        
                        for source in report["sources"]:
                            with st.expander(f"{source['title']}"):
                                st.write(f"**Authors:** {', '.join(source['authors'][:3])}...")
                                st.write(f"**Published:** {source['published']}")
                                st.write(f"**ID:** {source['paper_id']}")
                    
                    except Exception as e:
                        st.error(f"Error during research: {str(e)}")


if __name__ == "__main__":
    main()
