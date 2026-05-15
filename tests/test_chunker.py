import pytest
from papermind.ingestion.chunker import Chunker

def test_chunker_basic():
    chunker = Chunker(chunk_size=10, overlap=2)
    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15"
    metadata = {
        'id': 'test_id',
        'title': 'Test Title'
    }

    chunks = chunker.chunk_text(text, metadata)

    assert len(chunks) == 2

    # First chunk should have 10 words
    assert len(chunks[0]['text'].split()) == 10
    assert chunks[0]['text'] == "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
    assert chunks[0]['paper_id'] == 'test_id'

    # Second chunk should overlap by 2 words: word9 word10 ...
    assert len(chunks[1]['text'].split()) == 7  # the remaining words
    assert chunks[1]['text'] == "word9 word10 word11 word12 word13 word14 word15"

def test_chunk_papers():
    chunker = Chunker(chunk_size=5, overlap=1)
    papers = [
        {
            'id': 'paper1',
            'text': 'this is a test paper with some words'
        },
        {
            'id': 'paper2',
            'text': 'another test paper here with different words'
        }
    ]

    chunks = chunker.chunk_papers(papers)
    assert len(chunks) == 4

    assert chunks[0]['paper_id'] == 'paper1'
    assert chunks[2]['paper_id'] == 'paper2'
