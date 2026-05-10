"""
TextProcessor Class
Handles text preprocessing - implementation from notebook's light_clean_text function
"""

import re
from typing import List, Tuple, Set


class TextProcessor:
    """
    Text preprocessing and paragraph processing
    Based on notebook functions: light_clean_text, split_into_paragraphs, etc.
    """
    
    def __init__(self):
        """Initialize TextProcessor with stopwords and configuration"""
        self.stop_words: Set[str] = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
            'why', 'how', 'this', 'that', 'these', 'those', 'it', 'its',
            'of', 'in', 'on', 'at', 'by', 'for', 'with', 'to', 'from',
            'and', 'or', 'but', 'if', 'then', 'else', 'so', 'as', 'than'
        }
        self.min_paragraph_length: int = 100
    
    def preprocess_text(self, text: str) -> str:
        """
        Perform light cleaning on text.
        Implementation from notebook's light_clean_text function.
        """
        if not isinstance(text, str):
            return text
        
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
        
        return text
    
    def split_into_paragraphs(self, text: str, min_length: int = 50, 
                             min_paragraph_length: int = 100) -> List[str]:
        """
        Split text into paragraphs.
        Implementation from notebook's split_into_paragraphs function.
        """
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = text.split('\n')
        
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            stripped_line = line.strip()
            
            if not stripped_line:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                if current_paragraph:
                    current_text = ' '.join(current_paragraph)
                    if len(current_text) >= min_length:
                        paragraphs.append(current_text)
                        current_paragraph = [stripped_line]
                    else:
                        current_paragraph.append(stripped_line)
                else:
                    current_paragraph.append(stripped_line)
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        paragraphs = [
            p.strip() for p in paragraphs 
            if p.strip() and len(p.strip()) >= min_paragraph_length
        ]
        
        return paragraphs
    
    def select_best_paragraph(self, question: str, paragraphs: List[str]) -> Tuple[str, int]:
        """
        Select best paragraph based on word matching score.
        Implementation from notebook's select_best_paragraph function.
        """
        if len(paragraphs) == 1:
            return paragraphs[0], 0
        
        scores = []
        for i, para in enumerate(paragraphs):
            score = self.word_match_score(question, para)
            scores.append((score, i, para))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx, best_para = scores[0]
        
        return best_para, best_idx
    
    def tokenize_simple(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split by non-alphanumeric"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def word_match_score(self, question: str, paragraph: str) -> int:
        """Calculate word matching score between question and paragraph"""
        q_tokens = set(self.tokenize_simple(question))
        p_tokens = set(self.tokenize_simple(paragraph))
        
        q_tokens = q_tokens - self.stop_words
        p_tokens = p_tokens - self.stop_words
        
        overlap = q_tokens & p_tokens
        return len(overlap)
