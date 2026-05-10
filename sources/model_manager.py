"""
ModelManager Class
Manages RoBERTa model and tokenizer - implementation from notebook's load_model
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import os
from typing import Optional


class ModelManager:
    """
    RoBERTa model and tokenizer management.
    Based on notebook's load_model function and model loading logic.
    """
    
    def __init__(self):
        """Initialize ModelManager"""
        self.model: Optional[AutoModelForQuestionAnswering] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.qa_pipeline: Optional[pipeline] = None
    
    def load_model(self, model_path: str):
        """
        Load QA model and return pipeline.
        Implementation from notebook's load_model function (cached with @st.cache_resource).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        
        device = 0 if torch.cuda.is_available() else -1
        
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )
    
    def get_model(self) -> AutoModelForQuestionAnswering:
        """Get the loaded model"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model
    
    def get_tokenizer(self) -> AutoTokenizer:
        """Get the loaded tokenizer"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")
        return self.tokenizer
    
    def predict(self, question: str, context: str) -> dict:
        """
        Run prediction using QA pipeline.
        Implementation from notebook where qa_pipeline is called.
        """
        if self.qa_pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call load_model() first.")
        
        result = self.qa_pipeline(question=question, context=context)
        return result
