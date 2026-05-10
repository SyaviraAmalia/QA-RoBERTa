import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import os
import re
from collections import Counter

st.set_page_config(page_title="Q&A", page_icon="🎯")
st.title("🎯 Question Answering")


MODEL_PATH = "C:\\Users\\hp\\Documents\\SKRIPSI\\Project\\models\\scenario_3\\final_model"

@st.cache_resource
def load_model(model_path):
    """Load QA model dan return pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device, max_length=512, truncation=True)
    return qa_pipeline


# FILE EXTRACTION FUNCTIONS
def extract_text_from_txt(file):
    return file.read().decode('utf-8'), None

def extract_text_from_pdf(file):
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file)
        pages_text = []
        full_text = ""
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            pages_text.append({
                'page_num': page_num,
                'text': page_text,
                'start_char': len(full_text),
                'end_char': len(full_text) + len(page_text)
            })
            full_text += page_text + "\n"
        
        return full_text.strip(), pages_text
    except ImportError:
        st.error("❌ PyPDF2 tidak terinstall. Jalankan: `pip install PyPDF2`")
        return None, None

def extract_text_from_docx(file):
    try:
        from docx import Document
        doc = Document(file)
        
        full_text = ""
        for para in doc.paragraphs:
            if para.text.strip():
                full_text += para.text.strip() + "\n"
        
        CHARS_PER_PAGE = 3700
        text_length = len(full_text)
        estimated_pages = max(1, round(text_length / CHARS_PER_PAGE))
        
        pages_text = []
        chars_per_page = text_length / estimated_pages
        
        for page_num in range(1, estimated_pages + 1):
            start = int((page_num - 1) * chars_per_page)
            end = int(page_num * chars_per_page) if page_num < estimated_pages else text_length
            
            pages_text.append({
                'page_num': page_num,
                'text': full_text[start:end],
                'start_char': start,
                'end_char': end
            })
        
        return full_text.strip(), pages_text
    except ImportError:
        st.error("❌ python-docx tidak terinstall. Jalankan: `pip install python-docx`")
        return None, None


def split_into_chunks(text, chunk_size=512, overlap=128):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        if len(chunk_text.strip()) >= 100:
            chunks.append({
                'text': chunk_text.strip(),
                'start_word': i,
                'end_word': i + len(chunk_words),
                'index': len(chunks)
            })
    
    return chunks if chunks else [{'text': text, 'start_word': 0, 'end_word': len(words), 'index': 0}]


# WORD MATCHING FUNCTIONS
def tokenize_and_filter_stopwords(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    
    stopwords = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 
        'why', 'how', 'this', 'that', 'these', 'those', 'it', 'its',
        'of', 'in', 'on', 'at', 'by', 'for', 'with', 'to', 'from',
        'and', 'or', 'but', 'if', 'then', 'else', 'so', 'as', 'than',
        'yang', 'di', 'ke', 'dari', 'pada', 'untuk', 'dengan', 'dalam',
        'oleh', 'sebagai', 'adalah', 'ini', 'itu', 'atau', 'dan', 'juga'
    }
    
    filtered_tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    return filtered_tokens

def calculate_word_matching_score(question, chunk_text):
    question_tokens = set(tokenize_and_filter_stopwords(question))
    chunk_tokens = set(tokenize_and_filter_stopwords(chunk_text))
    
    matching_words = question_tokens & chunk_tokens
    score = len(matching_words)
    
    return score, list(matching_words)

def rank_chunks_by_word_matching(question, chunks):
    scored_chunks = []
    
    for chunk in chunks:
        score, matching_words = calculate_word_matching_score(question, chunk['text'])
        scored_chunks.append({
            'chunk': chunk,
            'score': score,
            'matching_words': matching_words
        })
    
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    return scored_chunks


def answer_question_hybrid(question, chunks, qa_pipeline, use_filtering=True):
    all_results = []
    
    # Strategy 1: Word Matching + Top-K
    if use_filtering:
        scored_chunks = rank_chunks_by_word_matching(question, chunks)
        
        # PERBAIKAN: Ambil top-10 (bukan 5) dan min_score=0 (bukan 1)
        top_k = min(10, len(chunks))
        selected_chunks = [sc['chunk'] for sc in scored_chunks[:top_k]]
        
        # Run QA on selected chunks
        for chunk in selected_chunks:
            try:
                result = qa_pipeline(question=question, context=chunk['text'])

                # 🔧 CLIPPING SCORE
                clipped_score = min(1.0, max(0.0, result['score']))

                all_results.append({
                    'answer': result['answer'],
                    'score': clipped_score,  # ✅ PAKAI clipped_score
                    'chunk_idx': chunk['index'],
                    'chunk_text': chunk['text'],
                    'method': 'word_matching'
                })
            except:
                continue
    
    # Strategy 2: Check semua chunks jika:
    # - Tidak ada hasil dari word matching, ATAU
    # - Best score < threshold
    need_full_search = False
    
    if not all_results:
        need_full_search = True
    elif max(r['score'] for r in all_results) < 0.35:
        need_full_search = True
    
    if need_full_search:
        for chunk in chunks:
            # Skip yang sudah di-check
            if any(r['chunk_idx'] == chunk['index'] for r in all_results):
                continue
            
            try:
                result = qa_pipeline(question=question, context=chunk['text'])

                # 🔧 CLIPPING SCORE
                clipped_score = min(1.0, max(0.0, result['score']))

                all_results.append({
                    'answer': result['answer'],
                    'score': clipped_score,  # ✅ PAKAI clipped_score
                    'chunk_idx': chunk['index'],
                    'chunk_text': chunk['text'],
                    'method': 'full_search'
                })
            except:
                continue
    
    if not all_results:
        return None, [], []  # ✅ RETURN 3 VALUES
    
    # Sort by score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    return all_results[0], all_results, []  # ✅ RETURN 3 VALUES


# PAGE LOCATION FUNCTIONS
def find_answer_page(original_text, answer, pages_info):
    import re
    
    def normalize_text(text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    normalized_doc = normalize_text(original_text)
    normalized_answer = normalize_text(answer)
    
    answer_pos = normalized_doc.find(normalized_answer)
    found_answer = normalized_answer
    
    if answer_pos == -1:
        answer_pos = normalized_doc.lower().find(normalized_answer.lower())
        if answer_pos != -1:
            found_answer = normalized_doc[answer_pos:answer_pos + len(normalized_answer)]
    
    if answer_pos == -1:
        answer_words = normalized_answer.split()
        if len(answer_words) > 3:
            partial = ' '.join(answer_words[:int(len(answer_words) * 0.7)])
            answer_pos = normalized_doc.lower().find(partial.lower())
            if answer_pos != -1:
                found_answer = partial
    
    if answer_pos == -1:
        return {
            'found': False,
            'original_answer': answer,
            'has_page_info': pages_info is not None
        }
    
    if pages_info:
        for page in pages_info:
            page_normalized = normalize_text(page['text'])
            
            if normalized_answer.lower() in page_normalized.lower():
                return {
                    'found': True,
                    'page_num': page['page_num'],
                    'total_pages': len(pages_info),
                    'answer_text': found_answer,
                    'original_answer': answer,
                    'has_page_info': True
                }
        
        for page in pages_info:
            if answer_pos >= page['start_char'] and answer_pos < page['end_char']:
                return {
                    'found': True,
                    'page_num': page['page_num'],
                    'total_pages': len(pages_info),
                    'answer_text': found_answer,
                    'original_answer': answer,
                    'has_page_info': True
                }
    
    percentage = (answer_pos / len(normalized_doc)) * 100 if len(normalized_doc) > 0 else 0
    
    return {
        'found': True,
        'answer_text': found_answer,
        'original_answer': answer,
        'percentage': percentage,
        'has_page_info': False
    }

def create_answer_snippet(text, answer, context_chars=250):
    import re
    
    def normalize_spaces(t):
        return re.sub(r'\s+', ' ', t)
    
    normalized_text = normalize_spaces(text)
    normalized_answer = normalize_spaces(answer)
    
    answer_pos = normalized_text.find(normalized_answer)
    
    if answer_pos == -1:
        answer_pos = normalized_text.lower().find(normalized_answer.lower())
        if answer_pos != -1:
            normalized_answer = normalized_text[answer_pos:answer_pos + len(normalized_answer)]
    
    if answer_pos == -1:
        answer_words = normalized_answer.split()
        if len(answer_words) > 2:
            partial = ' '.join(answer_words[:max(2, len(answer_words)//2)])
            answer_pos = normalized_text.lower().find(partial.lower())
            if answer_pos != -1:
                end_pos = answer_pos + len(partial)
                remaining = normalized_text[end_pos:end_pos+100]
                extra_words = remaining.split()[:len(answer_words)-len(partial.split())]
                normalized_answer = partial + ' ' + ' '.join(extra_words)
    
    if answer_pos == -1:
        return None
    
    start_pos = max(0, answer_pos - context_chars)
    end_pos = min(len(normalized_text), answer_pos + len(normalized_answer) + context_chars)
    
    snippet = normalized_text[start_pos:end_pos]
    
    prefix = "..." if start_pos > 0 else ""
    suffix = "..." if end_pos < len(normalized_text) else ""
    
    snippet_highlighted = snippet.replace(
        normalized_answer,
        f"**:red[{normalized_answer}]**"
    )
    
    return f"{prefix}{snippet_highlighted}{suffix}"


# LOAD MODEL
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model tidak ditemukan di `{MODEL_PATH}`")
    st.info("Pastikan folder model sudah di-copy ke lokasi yang benar.")
    st.stop()

with st.spinner("Loading model..."):
    qa_pipeline = load_model(MODEL_PATH)


# UI - INPUT
st.markdown("### 📄 Input Context")

input_method = st.radio(
    "Select the context input method:",
    ["📂 Upload Document", "✏️ Type Manual"],
    horizontal=True
)

context = ""
pages_info = None

if input_method == "📂 Upload Document":
    uploaded_file = st.file_uploader(
        "Upload document:",
        type=['txt', 'pdf', 'docx'],
        help="Supported formats: .txt, .pdf, .docx"
    )
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner("Reading document..."):
            if file_extension == 'txt':
                context, pages_info = extract_text_from_txt(uploaded_file)
            elif file_extension == 'pdf':
                context, pages_info = extract_text_from_pdf(uploaded_file)
            elif file_extension == 'docx':
                context, pages_info = extract_text_from_docx(uploaded_file)
        
        if context:
            st.success(f"✅ Document read successfully: {uploaded_file.name}")
            
            words = len(context.split())
            chars = len(context)

            if pages_info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Pages", len(pages_info))
                with col2:
                    st.metric("📤 Words", f"{words:,}")
                with col3:
                    st.metric("📏 Characters", f"{chars:,}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📤 Words", f"{words:,}")
                with col2:
                    st.metric("📏 Characters", f"{chars:,}")
            
            with st.expander("📖 View Document Content", expanded=False):
                st.text(context[:2000] + "..." if len(context) > 2000 else context)
else:
    context = st.text_area(
        "Context:",
        height=200,
        placeholder="Type paragraph/context here..."
    )

st.markdown("### ❓ Input Question")
question = st.text_input(
    "Question:",
    placeholder="Type your question here..."
)

if question.strip() and not question.strip().endswith('?'):
    question = question.strip() + '?'


# PREDICTION
if st.button("🔍 Get Answer", type="primary"):
    if context and context.strip() and question.strip():
        with st.spinner("Processing..."):
            # PERBAIKAN: Gunakan parameter chunking yang sama dengan File 2
            chunks = split_into_chunks(context, chunk_size=512, overlap=128)
            
            # PERBAIKAN: Gunakan hybrid approach
            best_result, all_results, debug_info = answer_question_hybrid(
            question, 
            chunks, 
            qa_pipeline,
            use_filtering=True
        )
            
            if not best_result:
                st.error("❌ Failed to get answer")
                st.stop()
        
        # ============================================
        # DISPLAY RESULTS
        # ============================================

        st.markdown("---")

        confidence = best_result['score']

        # 🎯 TAMBAHKAN LOGIKA THRESHOLD
        MINIMUM_CONFIDENCE = 0.01  # Threshold minimal untuk dianggap "found"

        if confidence < MINIMUM_CONFIDENCE:
            # ❌ CONFIDENCE TERLALU RENDAH
            st.markdown("### ❌ No Reliable Answer Found")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Attempt", best_result['answer'] if best_result['answer'] else "N/A")
            with col2:
                st.metric("Confidence Score", f"{confidence:.4f}")
            
            st.error(
                "⚠️ The model could not find a reliable answer to your question. "
                "This could mean:\n"
                "- The answer is not in the provided context\n"
                "- The question is ambiguous or unclear\n"
                "- The context doesn't contain relevant information"
            )
            
        else:
            # ✅ CONFIDENCE CUKUP TINGGI
            st.markdown("### ✅ Answer Found")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Jawaban", best_result['answer'])
            with col2:
                # Tambahkan indikator kualitas
                if confidence >= 0.7:
                    confidence_label = f"{confidence:.4f}"
                    confidence_color = "🟢"
                elif confidence >= 0.3:
                    confidence_label = f"{confidence:.4f}"
                    confidence_color = "🟡"
                else:
                    confidence_label = f"{confidence:.4f}"
                    confidence_color = "🟠"
                
                st.metric("Confidence Score", confidence_label)
            
            # Warning untuk confidence rendah tapi masih di atas threshold
            if confidence < 0.3:
                st.warning(
                    f"{confidence_color} **Low Confidence Warning:** "
                    "The answer might not be accurate. Please verify the result."
                )
            
            # ============================================
            # LOCATION
            # ============================================
            st.markdown("### 📍 Answer Location in Document")
            
            location_info = find_answer_page(context, best_result['answer'], pages_info)
            
            if location_info['found']:
                if location_info['has_page_info']:
                    st.info(
                        f"📄 Found on **Page {location_info['page_num']}** "
                        f"of {location_info['total_pages']} total pages"
                    )
                else:
                    st.info(f"📍 Found at approximately **{location_info['percentage']:.1f}%** through the document")

                st.markdown("**📖 Answer Context:**")
                snippet = create_answer_snippet(context, location_info['answer_text'], context_chars=250)
                
                if snippet:
                    st.markdown(f"> {snippet}")
                else:
                    snippet_from_chunk = create_answer_snippet(
                        best_result['chunk_text'], 
                        best_result['answer'], 
                        context_chars=250
                    )
                    if snippet_from_chunk:
                        st.markdown(f"> {snippet_from_chunk}")
            else:
                st.warning("⚠️ Answer location not found in original document")
                snippet_from_chunk = create_answer_snippet(
                    best_result['chunk_text'], 
                    best_result['answer'], 
                    context_chars=250
                )
                if snippet_from_chunk:
                    st.markdown("**Context from processed chunk:**")
                    st.markdown(f"> {snippet_from_chunk}")
                
    else:
        st.warning("⚠️ Please provide both Context and Question!")