import streamlit as st

st.set_page_config(
    page_title="RoBERTa QA - Tugas Akhir",
    page_icon="🤖",
    layout="wide"
)

st.title("RoBERTa Question Answering")

st.markdown("""
## Selamat Datang!

Aplikasi ini adalah hasil fine-tuning model RoBERTa untuk tugas Question Answering.

### 📌 Halaman yang Tersedia:

1. **Demo Model** - Coba model dengan memasukkan context (teks manual atau upload dokumen) dan question
2. **Pengujian Model** - Upload file JSON/CSV untuk pengujian batch dan mendapatkan metrik F1 & Exact Match

---
**Author:** Syavira Amalia  
**Project:** Tugas Akhir - RoBERTa QA Fine-Tuning
""")
