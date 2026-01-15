import os
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from datetime import datetime
import time


class ArxivLoader:
    """Loads papers from arXiv API based on query, category, and year filters."""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, download_dir: str = "papermind/data/raw_papers"):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        category: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[Dict]:
        """
        Search arXiv for papers matching query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            category: arXiv category filter (e.g., 'cs.AI', 'cs.LG')
            year: Year filter
            
        Returns:
            List of paper metadata dictionaries
        """
        search_query = query
        
        if category:
            search_query = f"cat:{category} AND {query}"
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        
        papers = self._parse_response(response.text)
        
        if year:
            papers = [p for p in papers if self._extract_year(p['published']) == year]
        
        return papers
    
    def _parse_response(self, xml_response: str) -> List[Dict]:
        """Parse arXiv API XML response into structured data."""
        root = ET.fromstring(xml_response)
        namespace = {'atom': 'http://www.w3.org/2005/Atom',
                     'arxiv': 'http://arxiv.org/schemas/atom'}
        
        papers = []
        for entry in root.findall('atom:entry', namespace):
            paper_id = entry.find('atom:id', namespace).text.split('/abs/')[-1]
            
            paper = {
                'id': paper_id,
                'title': entry.find('atom:title', namespace).text.strip().replace('\n', ' '),
                'summary': entry.find('atom:summary', namespace).text.strip().replace('\n', ' '),
                'authors': [author.find('atom:name', namespace).text 
                           for author in entry.findall('atom:author', namespace)],
                'published': entry.find('atom:published', namespace).text,
                'pdf_url': entry.find('atom:link[@title="pdf"]', namespace).attrib['href'],
                'categories': [cat.attrib['term'] 
                              for cat in entry.findall('atom:category', namespace)]
            }
            papers.append(paper)
        
        return papers
    
    def _extract_year(self, date_string: str) -> int:
        """Extract year from date string."""
        return datetime.fromisoformat(date_string.replace('Z', '+00:00')).year
    
    def download_pdf(self, paper: Dict) -> str:
        """
        Download PDF for a given paper.
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            Local file path to downloaded PDF
        """
        pdf_url = paper['pdf_url']
        paper_id = paper['id'].replace('/', '_')
        local_path = os.path.join(self.download_dir, f"{paper_id}.pdf")
        
        if os.path.exists(local_path):
            return local_path
        
        time.sleep(1)
        
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        return local_path
    
    def download_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Download PDFs for multiple papers and return updated metadata.
        
        Args:
            papers: List of paper metadata dictionaries
            
        Returns:
            Updated list with local_path field added
        """
        for paper in papers:
            try:
                local_path = self.download_pdf(paper)
                paper['local_path'] = local_path
            except Exception as e:
                print(f"Failed to download {paper['id']}: {e}")
                paper['local_path'] = None
        
        return papers
