import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import os
import re
from collections import Counter

st.set_page_config(page_title="Q&A", page_icon="ðŸŽ¯")
st.title("ðŸŽ¯ Question Answering")


# MODEL PATH - Langsung ke Best Model (Scenario 3)
MODEL_PATH = "C:\\Users\\hp\\Documents\\SKRIPSI\\Project\\models\\scenario_3\\final_model"

# Confidence thresholds
HIGH_CONFIDENCE = 0.5
LOW_CONFIDENCE = 0.3

@st.cache_resource
def load_model(model_path):
    """Load QA model dan return pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)
    return qa_pipeline

# ============================================
# HELPER FUNCTIONS
# ============================================
def extract_text_from_txt(file):
    """Extract text dari file .txt"""
    return file.read().decode('utf-8')

def extract_text_from_pdf(file):
    """Extract text dari file .pdf menggunakan PyPDF2"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        st.error("âŒ PyPDF2 tidak terinstall. Jalankan: `pip install PyPDF2`")
        return None

def extract_text_from_docx(file):
    """Extract text dari file .docx menggunakan python-docx"""
    try:
        from docx import Document
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except ImportError:
        st.error("âŒ python-docx tidak terinstall. Jalankan: `pip install python-docx`")
        return None

def split_into_chunks(text, chunk_size=512, overlap=128):
    """
    Split text menjadi chunks dengan overlap untuk QA.
    chunk_size: dalam jumlah kata (bukan token, untuk simplicity)
    overlap: jumlah kata yang overlap antar chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        if len(chunk_text.strip()) >= 100:  # minimum chunk length
            chunks.append({
                'text': chunk_text.strip(),
                'start_word': i,
                'end_word': i + len(chunk_words)
            })
    
    return chunks if chunks else [{'text': text, 'start_word': 0, 'end_word': len(words)}]

def answer_question_multi_chunk(question, chunks, qa_pipeline, top_k=3):
    """
    Jalankan QA di multiple chunks dan pilih jawaban terbaik.
    
    Args:
        question: pertanyaan user
        chunks: list of chunk dicts
        qa_pipeline: QA pipeline
        top_k: berapa chunks dengan score tertinggi yang akan di-proses
    
    Returns:
        best_result: dict dengan answer, score, dan chunk info
    """
    results = []
    
    # Jalankan QA di setiap chunk
    for i, chunk in enumerate(chunks):
        try:
            result = qa_pipeline(question=question, context=chunk['text'])
            results.append({
                'answer': result['answer'],
                'score': result['score'],
                'chunk_idx': i,
                'chunk_text': chunk['text'],
                'start_word': chunk['start_word'],
                'end_word': chunk['end_word']
            })
        except Exception as e:
            # Skip jika ada error
            continue
    
    if not results:
        return None
    
    # Sort berdasarkan confidence score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Return jawaban dengan confidence tertinggi
    return results[0], results[:top_k]  # best result + top candidates



# HELPER FUNCTIONS - DISPLAY
def find_answer_in_original_document(original_text, answer):
    """
    Temukan lokasi jawaban di dokumen asli.
    Return: dict dengan info posisi yang user-friendly.
    """
    # Cari posisi jawaban (exact match)
    answer_pos = original_text.find(answer)
    
    # Jika tidak ketemu, coba case-insensitive
    if answer_pos == -1:
        answer_lower = answer.lower()
        text_lower = original_text.lower()
        answer_pos = text_lower.find(answer_lower)
        
        if answer_pos != -1:
            # Ambil actual text dari dokumen
            answer = original_text[answer_pos:answer_pos + len(answer)]
    
    if answer_pos == -1:
        return {'found': False}
    
    # Hitung nomor paragraf (berdasarkan double newline)
    text_before_answer = original_text[:answer_pos]
    paragraph_num = text_before_answer.count('\n\n') + 1
    
    # Hitung nomor baris (berdasarkan single newline)
    line_num = text_before_answer.count('\n') + 1
    
    # Hitung persentase posisi di dokumen
    percentage = (answer_pos / len(original_text)) * 100
    
    return {
        'found': True,
        'char_position': answer_pos,
        'paragraph_num': paragraph_num,
        'line_num': line_num,
        'percentage': percentage,
        'answer_text': answer
    }

def create_answer_snippet(text, answer, context_chars=200):
    """
    Buat snippet teks di sekitar jawaban dengan konteks yang lebih luas.
    """
    # Cari posisi jawaban
    answer_pos = text.find(answer)
    
    # Jika tidak ketemu, coba case-insensitive
    if answer_pos == -1:
        answer_pos = text.lower().find(answer.lower())
        if answer_pos != -1:
            answer = text[answer_pos:answer_pos + len(answer)]
    
    if answer_pos == -1:
        return None
    
    # Hitung start dan end position untuk snippet
    start_pos = max(0, answer_pos - context_chars)
    end_pos = min(len(text), answer_pos + len(answer) + context_chars)
    
    # Ambil snippet
    snippet = text[start_pos:end_pos]
    
    # Tambahkan ellipsis jika terpotong
    prefix = "..." if start_pos > 0 else ""
    suffix = "..." if end_pos < len(text) else ""
    
    # Highlight jawaban dalam snippet
    snippet_highlighted = snippet.replace(
        answer,
        f"**:red[{answer}]**"
    )
    
    return f"{prefix}{snippet_highlighted}{suffix}"

# LOAD MODEL

if not os.path.exists(MODEL_PATH):
    st.error(f"âŒ Model tidak ditemukan di `{MODEL_PATH}`")
    st.info("Pastikan folder model sudah di-copy ke lokasi yang benar.")
    st.stop()

with st.spinner("Loading model..."):
    qa_pipeline = load_model(MODEL_PATH)


# MAIN - Input Context & Question

st.markdown("### ðŸ“„ Input Context")

# Pilihan input: Manual atau Upload
input_method = st.radio(
    "Select the context input method:",
    ["ðŸ“‚ Upload Document", "âœï¸ Type Manual"],
    horizontal=True
)

context = ""

if input_method == "ðŸ“‚ Upload Document":

    uploaded_file = st.file_uploader(
        "Upload document:",
        type=['txt', 'pdf', 'docx'],
        help="Supported formats: .txt, .pdf, .docx"
    )
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner("Reading document..."):
            if file_extension == 'txt':
                context = extract_text_from_txt(uploaded_file)
            elif file_extension == 'pdf':
                context = extract_text_from_pdf(uploaded_file)
            elif file_extension == 'docx':
                context = extract_text_from_docx(uploaded_file)
        
        if context:
            st.success(f"âœ… Document read successfully: {uploaded_file.name}")
            with st.expander("ðŸ“– View Document Content", expanded=False):
                st.text(context[:2000] + "..." if len(context) > 2000 else context)
else:
    context = st.text_area(
        "Context:",
        height=200,
        placeholder="Type paragraph/context here..."
    )

# Input question
st.markdown("### â“ Input Question")
question = st.text_input(
    "Question:",
    placeholder="Type your question here..."
)

# Auto-append tanda tanya jika tidak ada
if question.strip() and not question.strip().endswith('?'):
    question = question.strip() + '?'


# PREDIKSI 

if st.button("🔍 Get Answer", type="primary"):
    if context and context.strip() and question.strip():
        with st.spinner("Processing..."):
            # Split context menjadi chunks dengan overlap
            chunks = split_into_chunks(context, chunk_size=512, overlap=128)
            
            st.info(f"📚 Document split into {len(chunks)} chunks with overlap")
            
            # Jalankan QA di semua chunks
            best_result, top_candidates = answer_question_multi_chunk(
                question, 
                chunks, 
                qa_pipeline,
                top_k=3
            )
            
            if not best_result:
                st.error("❌ Failed to get answer")
                st.stop()
        
   
        # HASIL PREDIKSI
   
        st.markdown("---")
        st.markdown("### 📊 Hasil Prediksi")
        
        confidence = best_result['score']
        
        # Display jawaban dan confidence
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Jawaban", best_result['answer'])
        with col2:
            st.metric("Confidence Score", f"{confidence:.4f}")
        
      
        # LOKASI JAWABAN
      
        st.markdown("### 📍 Answer Location")
        
        location_info = find_answer_in_original_document(context, best_result['answer'])
        
        if location_info['found']:
            # Info lokasi dalam format yang mudah dibaca
            st.info(
                f"📄 Found in **Paragraph {location_info['paragraph_num']}** "
                f"(approximately **{location_info['percentage']:.1f}%** through the document)"
            )
            
            # Tampilkan konteks
            st.markdown("**Context:**")
            snippet = create_answer_snippet(context, location_info['answer_text'], context_chars=250)
            
            if snippet:
                st.markdown(f'<div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; border-left: 4px solid #ff4b4b;">{snippet}</div>', unsafe_allow_html=True)
            
        else:
            st.warning("⚠️ Answer location not found in document")
