import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import evaluate
import os
import re
import string

st.set_page_config(page_title="Answer Prediction", page_icon="📊", layout="wide")
st.title("📊 Answer Prediction")

# Model paths
MODEL_PATHS = {
    "Scenario 1 (LR=2e-5, BS=8)": "C:\\Users\\hp\\Documents\\SKRIPSI\\Project\\models\\scenario_1\\final_model",
    "Scenario 2 (LR=2e-5, BS=16)": "C:\\Users\\hp\\Documents\\SKRIPSI\\Project\\models\\scenario_2\\final_model",
    "Scenario 3 (LR=3e-5, BS=32)": "C:\\Users\\hp\\Documents\\SKRIPSI\\Project\\models\\scenario_3\\final_model",
}

@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)
    return qa_pipeline

@st.cache_resource
def load_squad_metric():
    """Load SQuAD metric dari library evaluate (sama dengan training)."""
    return evaluate.load("squad")

# ============================================
# METRIC FUNCTIONS (SQuAD Official via evaluate library)
# ============================================
def compute_metrics_squad(predictions, references, squad_metric):
    """
    Compute F1 and EM menggunakan library evaluate (sama dengan training).
    
    Args:
        predictions: list of {"id": str, "prediction_text": str}
        references: list of {"id": str, "answers": {"text": [...], "answer_start": [...]}}
        squad_metric: loaded evaluate metric
    
    Returns:
        dict dengan 'f1' dan 'exact_match'
    """
    results = squad_metric.compute(predictions=predictions, references=references)
    return {
        'f1': results['f1'],
        'exact_match': results['exact_match']
    }

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# ============================================
# SIDEBAR
# ============================================
st.sidebar.header("⚙️ Pengaturan")
selected_model = st.sidebar.selectbox("Pilih Skenario Model:", list(MODEL_PATHS.keys()))

# Load SQuAD metric
with st.spinner("Loading SQuAD metric..."):
    squad_metric = load_squad_metric()

# st.sidebar.success("✅ SQuAD metric loaded!")

# ============================================
# MAIN CONTENT
# ============================================
st.markdown("### 📁 Pilih Sumber Data")

data_source = st.radio(
    "Sumber data pengujian:",
    ["Upload File (CSV/JSON)", "SQuAD Validation Dataset"]
)

df = None

# ============================================
# OPSI 1: UPLOAD FILE
# ============================================
if data_source == "Upload File (CSV/JSON)":
    st.markdown("""
    **Format yang didukung:**
    - CSV dengan kolom: `context`, `question`, `answer`
    - JSON format SQuAD atau format simple
    """)
    
    uploaded_file = st.file_uploader("Upload file", type=['csv', 'json'])
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        
        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                # Tambahkan id jika tidak ada
                if 'id' not in df.columns:
                    df['id'] = [str(i) for i in range(len(df))]
                # Handle multiple answers dengan delimiter |||
                df['answer'] = df['answer'].apply(
                    lambda x: x.split('|||') if isinstance(x, str) and '|||' in x else ([x] if isinstance(x, str) else x)
                )
                
            elif file_name.endswith('.json'):
                import json
                data = json.load(uploaded_file)
                
                if 'data' in data:
                    # Format SQuAD
                    rows = []
                    for article in data['data']:
                        for para in article['paragraphs']:
                            context = para['context']
                            for qa in para['qas']:
                                all_answers = [ans['text'] for ans in qa['answers']]
                                answer_starts = [ans['answer_start'] for ans in qa['answers']]
                                rows.append({
                                    'id': qa.get('id', str(len(rows))),
                                    'context': context,
                                    'question': qa['question'],
                                    'answer': all_answers,
                                    'answer_start': answer_starts
                                })
                    df = pd.DataFrame(rows)
                else:
                    df = pd.DataFrame(data)
                    if 'id' not in df.columns:
                        df['id'] = [str(i) for i in range(len(df))]
                    if 'answer' in df.columns:
                        df['answer'] = df['answer'].apply(
                            lambda x: x if isinstance(x, list) else [x]
                        )
            
            st.success(f"✅ File berhasil di-upload: {file_name}")
            
            # Display preview
            st.markdown("### 📋 Preview Data")
            st.dataframe(df.head(10))
            st.info(f"Total data: {len(df)} baris")
            
        except Exception as e:
            st.error(f"❌ files are not allowed")
            st.stop()

# ============================================
# OPSI 2: LOAD SQUAD LANGSUNG
# ============================================
elif data_source == "SQuAD Validation Dataset":
    from datasets import load_dataset
    
    @st.cache_data
    def load_squad():
        squad = load_dataset("squad")
        return squad["validation"]
    
    with st.spinner("Loading SQuAD validation dataset..."):
        squad_val = load_squad()
    
    st.success(f"✅ Loaded {len(squad_val):,} samples dari SQuAD validation")
    
    # Convert ke DataFrame
    data = []
    for example in squad_val:
        data.append({
            'id': example['id'],
            'context': example['context'],
            'question': example['question'],
            'answer': example['answers']['text'],  # Already list
            'answer_start': example['answers']['answer_start']  # Untuk format evaluate
        })
    df = pd.DataFrame(data)

# ============================================
# COMMON: Preview dan Processing
# ============================================
if df is not None:
    # Display data info
    # st.markdown("### 📋 Data yang akan diuji")
    # st.dataframe(df.head(10))
    # st.info(f"Total data: {len(df)} baris")
    
    # Pilih jumlah sample
    num_samples = st.number_input(
        "Pilih jumlah sample data:",
        min_value=1,
        max_value=len(df),
        value=min(100, len(df)),  # Default 100 atau max jika kurang
        step=1
    )

    # Sampling data sesuai jumlah yang dipilih
    if num_samples < len(df):
        df_sample = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    else:
        df_sample = df.reset_index(drop=True)

    # Display preview sample
    # st.markdown(f"### 📋 Preview Data Sample ({len(df_sample)} baris)")
    # st.dataframe(df_sample.head(10))

    # Process button
    if st.button("🚀 Jalankan Pengujian", type="primary"):
        # Load model
        model_path = MODEL_PATHS[selected_model]
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model tidak ditemukan di {model_path}")
            st.stop()
        
        with st.spinner("Loading model..."):
            qa_pipeline = load_model(model_path)
        
        # Run predictions
        st.markdown("### ⏳ Memproses Prediksi...")
        progress_bar = st.progress(0)
        
        predictions_list = []
        references_list = []
        pred_texts = []
        confidences = []
        
        for i, row in df_sample.iterrows():
            # Run QA pipeline
            result = qa_pipeline(question=row['question'], context=row['context'])
            pred_text = result['answer']
            pred_texts.append(pred_text)
            confidences.append(result['score'])
            
            # Format untuk evaluate library (sama dengan training)
            predictions_list.append({
                "id": str(row['id']),
                "prediction_text": pred_text
            })
            
            # Format references
            answers = row['answer'] if isinstance(row['answer'], list) else [row['answer']]
            # answer_start: gunakan dari data jika ada, atau dummy [0] * len(answers)
            if 'answer_start' in row and row['answer_start']:
                answer_starts = row['answer_start'] if isinstance(row['answer_start'], list) else [row['answer_start']]
            else:
                answer_starts = [0] * len(answers)
            
            references_list.append({
                "id": str(row['id']),
                "answers": {
                    "text": answers,
                    "answer_start": answer_starts
                }
            })
            
            progress_bar.progress((i + 1) / len(df_sample))
        
        # Add predictions to dataframe
        df_sample['predicted_answer'] = pred_texts
        df_sample['confidence'] = confidences
        
        # Compute metrics menggunakan library evaluate (sama dengan training)
        metrics = compute_metrics_squad(predictions_list, references_list, squad_metric)
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Hasil Evaluasi Model")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F1 Score", f"{metrics['f1']:.2f}%")
        with col2:
            st.metric("Exact Match", f"{metrics['exact_match']:.2f}%")
        
        # Show prediction table
        st.markdown("### 📋 Detail Hasil Prediksi")
        
        # Add correct/wrong column (untuk visual, berdasarkan exact match)
        def check_correct(row):
            pred = row['predicted_answer']
            truths = row['answer'] if isinstance(row['answer'], list) else [row['answer']]
            # Cek apakah prediksi cocok dengan SALAH SATU jawaban valid (exact match)
            return '✅' if any(normalize_answer(pred) == normalize_answer(t) for t in truths) else '❌'

        df_sample['correct'] = df_sample.apply(check_correct, axis=1)
        
        # Hitung statistik
        correct_count = (df_sample['correct'] == '✅').sum()
        incorrect_count = (df_sample['correct'] == '❌').sum()
        
        st.markdown(f"**Statistik:** ✅ Benar: {correct_count} | ❌ Salah: {incorrect_count}")
        
        st.dataframe(
            df_sample[['question', 'answer', 'predicted_answer', 'confidence', 'correct']],
            use_container_width=True
        )
        
        # Download results
        csv_result = df_sample.to_csv(index=False)
        st.download_button(
            label="📥 Download Hasil Prediksi (CSV)",
            data=csv_result,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )
