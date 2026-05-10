# 🤖 RoBERTa Question Answering Fine-Tuning

Fine-tuning model RoBERTa untuk task Question Answering (Extractive QA) menggunakan dataset SQuAD v1.

## 📋 Deskripsi Project

Project ini adalah implementasi fine-tuning model **RoBERTa-base** untuk sistem tanya jawab berbasis konteks (extractive QA) menggunakan dataset **SQuAD v1**. Project ini mencakup:

- ✅ 6 skenario training dengan hyperparameter berbeda
- ✅ Comprehensive evaluation (EM, F1, Error Analysis)

## 🎯 Skenario Training

| Scenario | Learning Rate | Batch Size | Epochs |
|----------|--------------|------------|--------|
| Scenario 1 | 1e-5 | 8 | 3 |
| Scenario 2 | 2e-5 | 8 | 3 |
| Scenario 3 | 2e-5 | 16 | 3 |
| Scenario 4 | 3e-5 | 16 | 3 |
| Scenario 5 | 3e-5 | 32 | 3 |
| Scenario 6 | 4e-5 | 32 | 3 |

## 📚 Dataset

**SQuAD v1.1** (Stanford Question Answering Dataset)
- Training: 87,599 question-answer pairs
- Validation: 10,570 question-answer pairs
- Download otomatis via Hugging Face `datasets` library


## 👨‍💻 Author

**Syavira Amalia**
- Email: syaviraamalia53@gmail.com
- LinkedIn: www.linkedin.com/in/syavira-amalia-0104a227a

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face untuk `transformers` library
- Stanford NLP Group untuk SQuAD dataset
- Facebook AI Research untuk RoBERTa model

## 📧 Contact

Untuk pertanyaan atau masalah, silakan buka issue di repository ini atau hubungi via email.

---

**Happy Fine-Tuning! 🚀**
