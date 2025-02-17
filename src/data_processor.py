import PyPDF2
from typing import List
import io

class FinancialDataProcessor:
    def __init__(self):
        pass
        
    def read_pdf(self, file_obj) -> str:
        """Extract text from uploaded PDF financial statements."""
        text = ""
        pdf_reader = PyPDF2.PdfReader(file_obj)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks