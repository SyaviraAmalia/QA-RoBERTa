"""
DocumentProcessor Class
Handles document text extraction - implementation from notebook's extract_text functions
"""

from typing import Optional


class DocumentProcessor:
    """
    Document text extraction from various file formats.
    Based on notebook functions: extract_text_from_txt, extract_text_from_pdf, extract_text_from_docx
    """
    
    def extract_text_from_txt(self, file) -> str:
        """
        Extract text from .txt file
        Implementation from notebook's extract_text_from_txt function.
        """
        return file.read().decode('utf-8')
    
    def extract_text_from_pdf(self, file) -> Optional[str]:
        """
        Extract text from .pdf file using PyPDF2
        Implementation from notebook's extract_text_from_pdf function.
        """
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except ImportError:
            return None
    
    def extract_text_from_docx(self, file) -> Optional[str]:
        """
        Extract text from .docx file using python-docx
        Implementation from notebook's extract_text_from_docx function.
        """
        try:
            from docx import Document
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except ImportError:
            return None
