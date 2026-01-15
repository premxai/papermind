import os
from typing import List, Dict, Optional
from openai import OpenAI
from papermind.mcp.memory import Memory
from papermind.vectorstore.faiss_store import FAISSStore
from papermind.ingestion.embedder import Embedder


class MCPController:
    """
    Multi-agent controller that orchestrates specialized research agents.
    Implements MCP-style coordination for autonomous research.
    """
    
    def __init__(
        self,
        vector_store: FAISSStore,
        embedder: Embedder,
        llm_model: str = "gpt-4",
        api_key: Optional[str] = None
    ):
        """
        Initialize MCP controller.
        
        Args:
            vector_store: FAISS vector store instance
            embedder: Embedder instance
            llm_model: LLM model to use for agents
            api_key: OpenAI API key
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_model = llm_model
        self.memory = Memory()
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        
        self.agents = {}
    
    def register_agent(self, name: str, agent):
        """Register a research agent with the controller."""
        self.agents[name] = agent
    
    def retrieve_context(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieve relevant context from vector store for a query.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks
        """
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding, k=k)
        return results
    
    def execute_agent(self, agent_name: str, query: str, context: List[Dict]) -> Dict:
        """
        Execute a specific agent with query and context.
        
        Args:
            agent_name: Name of agent to execute
            query: User query
            context: Retrieved context chunks
            
        Returns:
            Agent output
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not registered")
        
        agent = self.agents[agent_name]
        result = agent.execute(query, context, self.client, self.llm_model)
        
        self.memory.add_agent_result(agent_name, result)
        return result
    
    def orchestrate(self, query: str, retrieval_k: int = 15) -> Dict:
        """
        Orchestrate all agents to produce comprehensive research output.
        
        Args:
            query: User research question
            retrieval_k: Number of chunks to retrieve
            
        Returns:
            Combined results from all agents
        """
        self.memory.add_query(query)
        self.memory.add_message('user', query)
        
        context = self.retrieve_context(query, k=retrieval_k)
        
        results = {}
        for agent_name in self.agents.keys():
            try:
                result = self.execute_agent(agent_name, query, context)
                results[agent_name] = result
            except Exception as e:
                results[agent_name] = {'error': str(e)}
        
        final_report = self._compile_report(query, results, context)
        
        self.memory.add_message('assistant', str(final_report))
        
        return final_report
    
    def _compile_report(self, query: str, agent_results: Dict, context: List[Dict]) -> Dict:
        """Compile final research report from agent outputs."""
        return {
            'query': query,
            'literature_review': agent_results.get('literature', {}),
            'methods_analysis': agent_results.get('methods', {}),
            'results_analysis': agent_results.get('results', {}),
            'critique': agent_results.get('critique', {}),
            'synthesis': agent_results.get('synthesis', {}),
            'sources': self._extract_sources(context),
            'num_sources': len(set([c.get('paper_id') for c in context if c.get('paper_id')]))
        }
    
    def _extract_sources(self, context: List[Dict]) -> List[Dict]:
        """Extract unique paper sources from context."""
        seen = set()
        sources = []
        
        for chunk in context:
            paper_id = chunk.get('paper_id')
            if paper_id and paper_id not in seen:
                seen.add(paper_id)
                sources.append({
                    'paper_id': paper_id,
                    'title': chunk.get('title'),
                    'authors': chunk.get('authors'),
                    'published': chunk.get('published')
                })
        
        return sources
    
    def get_memory(self) -> Memory:
        """Get controller memory."""
        return self.memory
    
    def clear_memory(self):
        """Clear controller memory."""
        self.memory.clear()
