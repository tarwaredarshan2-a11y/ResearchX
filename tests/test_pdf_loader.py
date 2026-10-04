from utils.pdf_loader import chunk_text, detect_section, normalize_text


def test_chunking_preserves_metadata():
    text = "Introduction\n" + ("This method evaluates a dataset. " * 80)
    chunks = chunk_text(text, paper="Paper A", paper_id="abc", page=3, section="Introduction", chunk_size=180, overlap=30)

    assert chunks
    assert chunks[0].paper == "Paper A"
    assert chunks[0].page == 3
    assert chunks[0].section == "Introduction"
    assert chunks[0].chunk_id


def test_detect_section_and_normalize_text():
    text = "  1 Introduction \n\n\n This paper studies retrieval.  "

    assert detect_section(text) == "Introduction"
    assert "\n\n\n" not in normalize_text(text)
