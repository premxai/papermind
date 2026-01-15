import os
from typing import Dict, Optional
import pypdf


class PDFParser:
    """Extracts text content from PDF files."""
    
    def __init__(self, processed_dir: str = "papermind/data/processed"):
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text = []
        
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        
        return '\n\n'.join(text)
    
    def parse_paper(self, paper: Dict) -> Optional[Dict]:
        """
        Parse a paper and extract its text content.
        
        Args:
            paper: Paper metadata dictionary with local_path
            
        Returns:
            Updated paper dictionary with text content
        """
        if not paper.get('local_path') or not os.path.exists(paper['local_path']):
            return None
        
        try:
            text = self.extract_text(paper['local_path'])
            paper['text'] = text
            paper['text_length'] = len(text)
            
            text_file = os.path.join(
                self.processed_dir,
                f"{paper['id'].replace('/', '_')}.txt"
            )
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
            paper['text_path'] = text_file
            
            return paper
        except Exception as e:
            print(f"Failed to parse {paper['id']}: {e}")
            return None
    
    def parse_papers(self, papers: list) -> list:
        """
        Parse multiple papers.
        
        Args:
            papers: List of paper metadata dictionaries
            
        Returns:
            List of papers with text content
        """
        parsed = []
        for paper in papers:
            result = self.parse_paper(paper)
            if result:
                parsed.append(result)
        
        return parsed
