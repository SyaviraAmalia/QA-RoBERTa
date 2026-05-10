# 🤖 RoBERTa Question Answering Fine-Tuning

Fine-tuning model RoBERTa untuk task Question Answering (Extractive QA) menggunakan dataset SQuAD v1.

## 📋 Deskripsi Project

Project ini adalah implementasi fine-tuning model **RoBERTa-base** untuk sistem tanya jawab berbasis konteks (extractive QA) menggunakan dataset **SQuAD v1**. Project ini mencakup:

- ✅ 6 skenario training dengan hyperparameter berbeda
- ✅ Comprehensive evaluation (EM, F1, Error Analysis)
- ✅ TensorBoard logging untuk monitoring
- ✅ Checkpoint management dan resume training
- ✅ Visualization dan comparison antar skenario
- ✅ Interactive notebooks untuk eksplorasi

## 🎯 Skenario Training

| Scenario | Learning Rate | Batch Size | Epochs |
|----------|--------------|------------|--------|
| Scenario 1 | 1e-5 | 8 | 3 |
| Scenario 2 | 2e-5 | 8 | 3 |
| Scenario 3 | 2e-5 | 16 | 3 |
| Scenario 4 | 3e-5 | 16 | 3 |
| Scenario 5 | 3e-5 | 32 | 3 |
| Scenario 6 | 4e-5 | 32 | 3 |

## 📁 Struktur Project
```
roberta-qa-project/
├── configs/                    # Configuration files
│   ├── base_config.yaml       # Base configuration
│   └── scenarios.yaml         # Training scenarios
├── src/                       # Source code
│   ├── __init__.py
│   ├── config.py             # Config management
│   ├── data_preprocessing.py # Data preprocessing
│   ├── trainer.py            # Training logic
│   ├── evaluator.py          # Evaluation & metrics
│   └── utils.py              # Utility functions
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_scenarios.ipynb
│   └── 03_evaluation_analysis.ipynb
├── scripts/                   # Automation scripts
│   ├── setup_project.py
│   └── generate_report.py
├── experiments/               # Training experiments
│   ├── scenario_1/
│   ├── scenario_2/
│   └── ...
├── data/                     # Datasets
│   ├── raw/
│   └── processed/
├── logs/                     # Logs & TensorBoard
│   └── tensorboard/
├── models/                   # Saved models
│   └── final/
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🚀 Setup & Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd roberta-qa-project
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Project Structure
```bash
python scripts/setup_project.py
```

## 💻 Usage

### Option 1: Interactive Training (Recommended)

Gunakan Jupyter notebooks untuk training interaktif:
```bash
jupyter notebook
```

Buka notebook:
1. `01_data_exploration.ipynb` - Eksplorasi dataset
2. `02_training_scenarios.ipynb` - Training per skenario
3. `03_evaluation_analysis.ipynb` - Evaluasi dan analisis

### Option 2: Training via Python Script
```python
from src import QATrainer, load_config, load_scenario_config
from src.data_preprocessing import prepare_datasets
from transformers import AutoTokenizer

# Load config
base_config = load_config()
scenario_config = load_scenario_config(scenario_id=1)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Prepare datasets
train_ds, eval_ds = prepare_datasets(tokenizer, base_config)

# Initialize trainer
trainer = QATrainer(
    model_name="roberta-base",
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    config=scenario_config,
    output_dir="experiments/scenario_1"
)

# Train
results = trainer.train()
```

### Resume Training from Checkpoint
```python
# Resume training
results = trainer.train(resume_from_checkpoint="experiments/scenario_1/checkpoint-1000")
```

## 📊 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=logs/tensorboard
```

Buka browser: `http://localhost:6006`

## 📈 Evaluation

### Run Evaluation
```python
from src import QAEvaluator
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_dataset

# Load model
model = AutoModelForQuestionAnswering.from_pretrained("experiments/scenario_1")
tokenizer = AutoTokenizer.from_pretrained("experiments/scenario_1")

# Load dataset
raw_dataset = load_dataset("squad")["validation"]

# Initialize evaluator
evaluator = QAEvaluator(
    model=model,
    tokenizer=tokenizer,
    eval_dataset=eval_features,
    raw_dataset=raw_dataset,
    output_dir="experiments/scenario_1/evaluation"
)

# Evaluate
results = evaluator.evaluate()
print(f"F1 Score: {results['metrics']['f1']:.4f}")
print(f"Exact Match: {results['metrics']['exact_match']:.4f}")

# Error analysis
error_analysis = evaluator.analyze_errors()
```

## 📋 Results

### Comparison Table

| Scenario | Learning Rate | Batch Size | F1 Score | Exact Match | Training Time |
|----------|--------------|------------|----------|-------------|---------------|
| Scenario 1 | 1e-5 | 8 | TBD | TBD | TBD |
| Scenario 2 | 2e-5 | 8 | TBD | TBD | TBD |
| Scenario 3 | 2e-5 | 16 | TBD | TBD | TBD |
| Scenario 4 | 3e-5 | 16 | TBD | TBD | TBD |
| Scenario 5 | 3e-5 | 32 | TBD | TBD | TBD |
| Scenario 6 | 4e-5 | 32 | TBD | TBD | TBD |

*TBD: To Be Determined (setelah training selesai)*

## 🛠️ Hardware Requirements

### Minimum Requirements
- **CPU**: Intel i5 / AMD Ryzen 5
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU dengan 6 GB VRAM (GTX 1060 atau lebih tinggi)
- **Storage**: 50 GB free space

### Recommended Requirements
- **CPU**: Intel i7 / AMD Ryzen 7
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU dengan 8+ GB VRAM (RTX 3060 atau lebih tinggi)
- **Storage**: 100 GB SSD

### Cloud Options
- Google Colab (Free/Pro)
- Kaggle Notebooks (Free)
- AWS SageMaker
- Azure ML

## 📚 Dataset

**SQuAD v1.1** (Stanford Question Answering Dataset)
- Training: 87,599 question-answer pairs
- Validation: 10,570 question-answer pairs
- Download otomatis via Hugging Face `datasets` library

## 🔧 Troubleshooting

### Out of Memory (OOM) Error

**Solution 1**: Reduce batch size
```yaml
per_device_train_batch_size: 4  # Reduce dari 8
gradient_accumulation_steps: 4   # Increase untuk kompensasi
```

**Solution 2**: Enable gradient checkpointing
```yaml
gradient_checkpointing: true
```

**Solution 3**: Use mixed precision
```yaml
fp16: true  # Untuk NVIDIA GPU
```

### Slow Training

**Solution**: 
- Increase `dataloader_num_workers`
- Enable `dataloader_pin_memory`
- Use SSD untuk data storage
- Reduce `logging_steps` dan `eval_steps`

## 📖 References

1. **RoBERTa**: [Liu et al., 2019](https://arxiv.org/abs/1907.11692)
2. **SQuAD**: [Rajpurkar et al., 2016](https://arxiv.org/abs/1606.05250)
3. **Hugging Face Transformers**: [Documentation](https://huggingface.co/docs/transformers/)

## 👨‍💻 Author

**[Your Name]**
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

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