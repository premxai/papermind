import pytest
from unittest.mock import patch, MagicMock
from papermind.ingestion.arxiv_loader import ArxivLoader

@patch('papermind.ingestion.arxiv_loader.requests.get')
def test_arxiv_loader_search(mock_get):
    # Mocking arXiv response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2101.00001</id>
    <updated>2021-01-01T00:00:00Z</updated>
    <published>2021-01-01T00:00:00Z</published>
    <title>Test Paper</title>
    <summary>This is a test summary.</summary>
    <author><name>John Doe</name></author>
    <link href="http://arxiv.org/pdf/2101.00001" title="pdf" type="application/pdf"/>
    <category term="cs.AI"/>
  </entry>
</feed>
    """
    mock_get.return_value = mock_response

    loader = ArxivLoader(download_dir='/tmp/test_dir')
    papers = loader.search(query='test', max_results=1)

    assert len(papers) == 1
    assert papers[0]['title'] == 'Test Paper'
    assert papers[0]['id'] == '2101.00001'
    assert papers[0]['authors'] == ['John Doe']
    assert papers[0]['pdf_url'] == 'http://arxiv.org/pdf/2101.00001'

@patch('papermind.ingestion.arxiv_loader.requests.get')
def test_arxiv_loader_retry_on_429(mock_get):
    # Mock rate limit and then success
    mock_429 = MagicMock()
    mock_429.status_code = 429

    mock_200 = MagicMock()
    mock_200.status_code = 200
    mock_200.text = """<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""

    # We need to raise HTTPError when status_code is 429
    import requests
    def raise_for_status_429():
        raise requests.exceptions.HTTPError(response=mock_429)

    mock_429.raise_for_status.side_effect = raise_for_status_429

    # First call returns 429, second returns 200
    mock_get.side_effect = [mock_429, mock_200]

    with patch('papermind.ingestion.arxiv_loader.time.sleep') as mock_sleep:
        loader = ArxivLoader(download_dir='/tmp/test_dir')
        papers = loader.search(query='test', max_results=1)

    assert mock_get.call_count == 2
    mock_sleep.assert_called_once()
