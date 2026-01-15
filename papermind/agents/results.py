from typing import List, Dict


class ResultsAgent:
    """Analyzes and compares experimental results across papers."""
    
    SYSTEM_PROMPT = """You are a research results analyst. Your task is to:
1. Extract and compare experimental results from papers
2. Identify performance metrics and benchmarks used
3. Analyze trends in empirical findings
4. Compare effectiveness of different approaches

Focus on quantitative results, datasets, and empirical evidence."""
    
    def execute(self, query: str, context: List[Dict], client, model: str) -> Dict:
        """
        Execute results analysis.
        
        Args:
            query: Research question
            context: Retrieved context chunks
            client: OpenAI client
            model: LLM model name
            
        Returns:
            Results analysis
        """
        context_text = self._format_context(context)
        
        user_message = f"""Research Question: {query}

Relevant Papers:
{context_text}

Provide a comprehensive results analysis covering:
1. Key experimental findings
2. Performance comparisons across approaches
3. Benchmarks and datasets used
4. Trends in empirical results"""
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3
        )
        
        return {
            'agent': 'results',
            'analysis': response.choices[0].message.content,
            'sources_used': len(set([c.get('paper_id') for c in context if c.get('paper_id')]))
        }
    
    def _format_context(self, context: List[Dict]) -> str:
        """Format context chunks for prompt."""
        formatted = []
        
        for chunk in context:
            text = chunk.get('text', '')
            if any(keyword in text.lower() for keyword in ['result', 'experiment', 'performance', 'accuracy', 'benchmark', 'dataset']):
                formatted.append(
                    f"Paper: {chunk.get('title')}\n"
                    f"Content: {text[:800]}\n"
                )
        
        if not formatted:
            for chunk in context[:10]:
                formatted.append(
                    f"Paper: {chunk.get('title')}\n"
                    f"Content: {chunk.get('text', '')[:800]}\n"
                )
        
        return '\n---\n'.join(formatted[:10])
