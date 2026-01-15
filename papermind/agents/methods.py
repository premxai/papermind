from typing import List, Dict


class MethodsAgent:
    """Analyzes and compares research methods across papers."""
    
    SYSTEM_PROMPT = """You are a research methods analyst. Your task is to:
1. Identify the methodologies used in relevant papers
2. Compare different approaches and techniques
3. Analyze strengths and weaknesses of each method
4. Identify best practices and methodological innovations

Focus on technical details and empirical approaches."""
    
    def execute(self, query: str, context: List[Dict], client, model: str) -> Dict:
        """
        Execute methods analysis.
        
        Args:
            query: Research question
            context: Retrieved context chunks
            client: OpenAI client
            model: LLM model name
            
        Returns:
            Methods analysis results
        """
        context_text = self._format_context(context)
        
        user_message = f"""Research Question: {query}

Relevant Papers:
{context_text}

Provide a comprehensive methods analysis covering:
1. Key methodologies employed
2. Comparison of different approaches
3. Strengths and limitations of each method
4. Methodological innovations and best practices"""
        
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
            'agent': 'methods',
            'analysis': response.choices[0].message.content,
            'sources_used': len(set([c.get('paper_id') for c in context if c.get('paper_id')]))
        }
    
    def _format_context(self, context: List[Dict]) -> str:
        """Format context chunks for prompt."""
        formatted = []
        
        for chunk in context:
            text = chunk.get('text', '')
            if any(keyword in text.lower() for keyword in ['method', 'approach', 'algorithm', 'technique', 'experiment']):
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
