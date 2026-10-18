import pytest
from pypdf import PdfWriter

from notes_loader.load_notes import load_notes, NoTextInPDF


def test_pdf_without_text_stops_before_any_llm_call(tmp_path):
    pdf = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)        # like a scan: a page without any text
    writer.write(pdf)
    with pytest.raises(NoTextInPDF):
        load_notes(str(pdf), "test_student")
