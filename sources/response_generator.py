"""
ResponseGenerator Class
Generates responses with answer location and snippets - implementation from notebook
"""

from typing import Tuple, Optional, Dict
from model_manager import ModelManager
from text_processor import TextProcessor
from document_processor import DocumentProcessor


class ResponseGenerator:
    """
    Response generation for question answering.
    Based on notebook's prediction logic, find_answer_location, and create_answer_snippet functions.
    """
    
    def __init__(self, model_manager: ModelManager, 
                 text_processor: TextProcessor,
                 document_processor: DocumentProcessor):
        """Initialize ResponseGenerator"""
        self.model_manager = model_manager
        self.text_processor = text_processor
        self.document_processor = document_processor
    
    def get_response(self, question: str, context: str) -> Tuple[dict, str, int]:
        """
        Get response for a question given context.
        Implementation from notebook's main prediction flow.
        """
        paragraphs = self.text_processor.split_into_paragraphs(context)
        
        if len(paragraphs) > 1:
            selected_paragraph, para_idx = self.text_processor.select_best_paragraph(
                question, paragraphs
            )
        else:
            selected_paragraph = context
            para_idx = 0
        
        result = self.model_manager.predict(question, selected_paragraph)
        
        return result, selected_paragraph, para_idx
    
    def find_answer_location(self, text: str, answer: str,
                            para_idx: Optional[int] = None,
                            total_paras: Optional[int] = None) -> Optional[Dict]:
        """
        Find location of answer in text.
        Implementation from notebook's find_answer_location function.
        """
        location_info = {}
        
        if para_idx is not None and total_paras is not None:
            location_info['paragraph'] = para_idx + 1
            location_info['total_paragraphs'] = total_paras
        
        answer_pos = text.find(answer)
        if answer_pos == -1:
            return None
        
        location_info['char_position'] = answer_pos
        
        return location_info
    
    def create_answer_snippet(self, text: str, answer: str, 
                             context_chars: int = 100) -> Optional[str]:
        """
        Create snippet of text around answer with highlighting.
        Implementation from notebook's create_answer_snippet function.
        """
        answer_pos = text.find(answer)
        if answer_pos == -1:
            return None
        
        start_pos = max(0, answer_pos - context_chars)
        end_pos = min(len(text), answer_pos + len(answer) + context_chars)
        
        snippet = text[start_pos:end_pos]
        
        prefix = "..." if start_pos > 0 else ""
        suffix = "..." if end_pos < len(text) else ""
        
        snippet_highlighted = snippet.replace(
            answer,
            f"**:green[{answer}]**"
        )
        
        return f"{prefix}{snippet_highlighted}{suffix}"
