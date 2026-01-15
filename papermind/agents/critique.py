from typing import List, Dict


class CritiqueAgent:
    """Identifies limitations, weaknesses, and critical issues in research."""
    
    SYSTEM_PROMPT = """You are a research critic. Your task is to:
1. Identify limitations in existing research approaches
2. Highlight methodological weaknesses
3. Point out gaps in evaluation or validation
4. Suggest areas that need further investigation

Be constructive but thorough in identifying issues."""
    
    def execute(self, query: str, context: List[Dict], client, model: str) -> Dict:
        """
        Execute critical analysis.
        
        Args:
            query: Research question
            context: Retrieved context chunks
            client: OpenAI client
            model: LLM model name
            
        Returns:
            Critical analysis results
        """
        context_text = self._format_context(context)
        
        user_message = f"""Research Question: {query}

Relevant Papers:
{context_text}

Provide a critical analysis covering:
1. Key limitations in current approaches
2. Methodological weaknesses
3. Gaps in validation or evaluation
4. Unresolved challenges and open problems"""
        
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
            'agent': 'critique',
            'analysis': response.choices[0].message.content,
            'sources_used': len(set([c.get('paper_id') for c in context if c.get('paper_id')]))
        }
    
    def _format_context(self, context: List[Dict]) -> str:
        """Format context chunks for prompt."""
        formatted = []
        
        for chunk in context:
            text = chunk.get('text', '')
            if any(keyword in text.lower() for keyword in ['limitation', 'weakness', 'challenge', 'future work', 'issue']):
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
