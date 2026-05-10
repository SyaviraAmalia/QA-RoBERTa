"""
UIManager Class
Manages Streamlit UI components - implementation from notebook's display functions
"""

import streamlit as st
from typing import Optional, Dict


class UIManager:
    """
    UI management for Streamlit application.
    Based on notebook's display functions and UI logic.
    """
    
    def __init__(self, styles: str = ""):
        """Initialize UIManager"""
        self.styles = styles
        self.HIGH_CONFIDENCE = 0.5
        self.LOW_CONFIDENCE = 0.3
    
    def display_qa_interface(self):
        """
        Display question answering interface.
        Implementation from notebook's main UI section.
        """
        st.markdown("### 📄 Input Context")
        
        input_method = st.radio(
            "Select the context input method:",
            ["✏️ Type Manual", "📁 Upload Document"],
            horizontal=True
        )
        
        context = ""
        uploaded_file = None
        
        if input_method == "✏️ Type Manual":
            context = st.text_area(
                "Context:",
                height=200,
                placeholder="Type paragraph/context here..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload document:",
                type=['txt', 'pdf', 'docx'],
                help="Supported formats: .txt, .pdf, .docx"
            )
        
        st.markdown("### ❓ Input Question")
        question = st.text_input(
            "Question:",
            placeholder="Type your question here..."
        )
        
        if question.strip() and not question.strip().endswith('?'):
            question = question.strip() + '?'
        
        return context, question, input_method, uploaded_file
    
    def display_benchmark_interface(self):
        """Display benchmark interface"""
        st.markdown("### 📊 Model Benchmark")
        st.info("Benchmark interface")
    
    def render_message(self, message: str):
        """Render message to user"""
        st.info(message)
    
    def handle_user_input(self) -> str:
        """Handle user text input"""
        return st.text_input("User input:")
    
    def display_confidence_status(self, confidence: float):
        """
        Display status based on confidence score.
        Implementation from notebook's display_confidence_status function.
        """
        if confidence >= self.HIGH_CONFIDENCE:
            st.success(f"✅ Confidence tinggi ({confidence:.2%}) - Jawaban reliable")
        elif confidence >= self.LOW_CONFIDENCE:
            st.warning(f"⚠️ Confidence sedang ({confidence:.2%}) - Jawaban perlu diverifikasi")
        else:
            st.error(f"❌ Confidence rendah ({confidence:.2%}) - Jawaban kemungkinan tidak ditemukan dalam konteks")
    
    def display_answer_location(self, location_info: Dict):
        """
        Display answer location information.
        Implementation from notebook's answer location display section.
        """
        st.markdown("### 📍 Answer Location")
        
        if 'paragraph' in location_info:
            st.markdown(f"**Paragraf:** {location_info['paragraph']} dari {location_info['total_paragraphs']}")
        else:
            st.markdown("**Paragraf:** 1 dari 1")
