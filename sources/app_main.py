"""
APP Class
Main application orchestrator - implementation from 1_Ask_Question.py main flow
"""

import streamlit as st
import os
from model_manager import ModelManager
from text_processor import TextProcessor
from document_processor import DocumentProcessor
from response_generator import ResponseGenerator
from ui_manager import UIManager
from benchmark_processor import BenchmarkProcessor


class APP:
    """
    Main application class that coordinates all components.
    Based on 1_Ask_Question.py main execution flow.
    """
    
    def __init__(self, model_path: str):
        """Initialize APP with all components"""
        self.model_path = model_path
        
        self.ui_manager = UIManager()
        self.model_manager = ModelManager()
        self.text_processor = TextProcessor()
        self.document_processor = DocumentProcessor()
        self.benchmark_processor = BenchmarkProcessor()
        
        self.response_generator = ResponseGenerator(
            self.model_manager,
            self.text_processor,
            self.document_processor
        )
    
    def run(self):
        """Run the main application - implementation from 1_Ask_Question.py"""
        st.set_page_config(page_title="Document Q&A", page_icon="🎯")
        st.title("🎯 Document Q&A")
        
        if not os.path.exists(self.model_path):
            st.error(f"❌ Model tidak ditemukan di `{self.model_path}`")
            st.info("Pastikan folder model sudah di-copy ke lokasi yang benar.")
            st.stop()
        
        with st.spinner("Loading model..."):
            self.model_manager.load_model(self.model_path)
        
        context, question, input_method, uploaded_file = self.ui_manager.display_qa_interface()
        
        if input_method == "📁 Upload Document" and uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            with st.spinner("Reading document..."):
                if file_extension == 'txt':
                    context = self.document_processor.extract_text_from_txt(uploaded_file)
                elif file_extension == 'pdf':
                    context = self.document_processor.extract_text_from_pdf(uploaded_file)
                elif file_extension == 'docx':
                    context = self.document_processor.extract_text_from_docx(uploaded_file)
            
            if context:
                st.success(f"✅ Document read successfully: {uploaded_file.name}")
                with st.expander("📖 View Document Content", expanded=False):
                    preview_text = context[:2000] + "..." if len(context) > 2000 else context
                    st.text(preview_text)
            else:
                st.error("❌ Failed to read document")
                return
        
        if st.button("🔍 Get Answer", type="primary"):
            if context and context.strip() and question.strip():
                with st.spinner("Processing..."):
                    result, selected_paragraph, para_idx = self.response_generator.get_response(
                        question, context
                    )
                    
                    paragraphs = self.text_processor.split_into_paragraphs(context)
                    total_paras = len(paragraphs)
                    
                    if total_paras > 1:
                        st.info(f"📚 Document has {total_paras} paragraphs. Selected paragraph #{para_idx + 1}")
                
                st.markdown("---")
                st.markdown("### 📊 Hasil Prediksi")
                
                confidence = result['score']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Jawaban", result['answer'])
                with col2:
                    st.metric("Confidence Score", f"{confidence:.4f}")
                
                self.ui_manager.display_confidence_status(confidence)
                
                location_info = self.response_generator.find_answer_location(
                    selected_paragraph,
                    result['answer'],
                    para_idx=para_idx if total_paras > 1 else None,
                    total_paras=total_paras if total_paras > 1 else None
                )
                
                if location_info:
                    self.ui_manager.display_answer_location(location_info)
                    
                    st.markdown("**Konteks Jawaban:**")
                    snippet = self.response_generator.create_answer_snippet(
                        selected_paragraph,
                        result['answer'],
                        context_chars=120
                    )
                    if snippet:
                        st.markdown(f'"{snippet}"')
                    else:
                        st.warning("⚠️ Tidak dapat membuat snippet jawaban")
                else:
                    st.warning("⚠️ Jawaban tidak ditemukan dalam konteks")
                
            else:
                st.warning("⚠️ Mohon isi Context dan Question terlebih dahulu!")
        
        st.markdown("---")
        with st.expander("ℹ️ Tentang Confidence Score"):
            st.markdown("""
            **Confidence Score** menunjukkan seberapa yakin model terhadap jawaban yang diberikan:
            
            - **≥ 50%**: Confidence tinggi - Jawaban reliable
            - **30% - 50%**: Confidence sedang - Jawaban perlu diverifikasi  
            - **< 30%**: Confidence rendah - Jawaban kemungkinan tidak ada dalam konteks
            
            *Score ini merupakan hasil softmax dari start/end position logits model.*
            """)


if __name__ == "__main__":
    MODEL_PATH = "C:\\Users\\hp\\Documents\\SKRIPSI\\Project\\models\\scenario_3\\final_model"
    app = APP(MODEL_PATH)
    app.run()
