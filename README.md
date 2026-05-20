# CMPE 346 - Multilingual Sentiment Analysis (Assignment 02)

This repository contains the implementation of a high-performance **Transformer-based Multilingual Sentiment Analysis model** designed to classify multi-lingual text reviews into binary sentiment categories (`positive` or `negative`).

---

## 🎯 Project Overview & Objectives
The goal of this project is to build and fine-tune a state-of-the-art multilingual sentiment classifier that:
1. Handles multiple languages (including English, Japanese, Chinese, French, Russian, German, Spanish, etc.) seamlessly using a single unified model.
2. Achieves a high validation F1-score well exceeding the class baseline of **0.7958**.
3. Implements clean, reusable, and robust `Preprocessor`, `Tokenizer`, and `Model` classes.
4. Generates serialized objects (`tokenizer.pkl`, `embedder.pkl`, and `model.pkl`) that are 100% compatible with the grading scripts.
5. Publishes the fine-tuned model publicly to the Hugging Face Model Hub.

---

## 🚀 Model Architecture & Checkpoint Choice

We selected **`xlm-roberta-base`** (270M parameters) as our foundation model. 

### Why XLM-RoBERTa?
* **Unsupervised Multilingual Pre-training:** Pre-trained on a massive 2.5TB dataset covering 100 distinct languages.
* **State-of-the-Art Vocabulary:** Utilizes a SentencePiece-based multilingual tokenizer with a large vocabulary (250k tokens), avoiding out-of-vocabulary (OOV) issues across languages like Japanese, Chinese, and Russian.
* **Proven Performance:** Demonstrates outstanding cross-lingual transferability and zero-shot performance on sequence classification tasks compared to mBERT.

---

## 📂 Project Structure

```text
├── data/
│   ├── train.csv                # Training dataset (140,000 samples)
│   └── valid.csv                # Validation dataset (17,500 samples)
├── main.py                      # End-to-end training pipeline
├── model.py                     # Model class wrapping HF SequenceClassifier
├── preprocessing.py             # Preprocessor, Tokenizer, and Embedder classes
├── run_test.py                  # Evaluation script (with fixed Preprocessor import)
├── saved_objects/               # Pickled artifacts required for evaluation
│   ├── tokenizer.pkl
│   ├── embedder.pkl
│   └── model.pkl
├── saved_model/                 # Raw Hugging Face model and tokenizer weights
└── README.md                    # Documentation
```

---

## 🛠️ Components & Implementation Details

### 1. Preprocessing (`preprocessing.py`)
* **`Preprocessor` Class:** Handles input pandas DataFrames by cleaning whitespace, managing potential missing values (NaNs), and formatting columns perfectly for the training and evaluation scripts.
* **`LabelEncoder` Class:** Encodes textual labels (`negative`, `positive`) into integers (`0`, `1`) and decodes them back. Left completely unchanged as per assignment rules.
* **`Tokenizer` Class:** Wraps the `xlm-roberta-base` tokenizer. We implemented **custom pickling hooks (`__getstate__` and `__setstate__`)** to exclude live, non-picklable C++ tokenizer objects, enabling 100% safe and cross-platform serialization under `tokenizer.pkl`.

### 2. Model Structure (`model.py`)
* **`Model` Class:** Wraps Hugging Face's `AutoModelForSequenceClassification`.
* **Flexible Inference (`predict`):** Dynamically supports multi-format input prediction:
  - Raw list of strings (tokenizes on-the-fly).
  - Hugging Face dataset dictionary inputs.
  - Raw list of input IDs (integers).
* **Robust Serialization:** Custom serialization state dictionary logic is implemented. It isolates the model weights from device-specific wrappers, preventing CPU/GPU/MPS device mapping conflicts when loaded on other machines.

### 3. Balanced Stratified Fine-Tuning (`main.py`)
To train the model efficiently while maintaining maximum generalization performance:
* We loaded the full training set of 140,000 samples.
* We created a **stratified, perfectly balanced training subset** of 20,000 samples (10,000 positive, 10,000 negative) to maintain class ratio and speed up training.
* We set the tokenization sequence length to `128` to capture the core sentiment of short reviews while accelerating training speed 4x on GPU.

---

## ⚙️ Hyperparameters

| Hyperparameter | Value |
| :--- | :--- |
| **Foundation Model** | `xlm-roberta-base` |
| **Learning Rate** | `2e-5` |
| **Optimizer** | `AdamW` |
| **Weight Decay** | `0.01` |
| **Batch Size** | `32` |
| **Epochs** | `2` |
| **Max Sequence Length** | `128` |

---

## 📊 Evaluation & Performance Results

Here are the results evaluated on the validation dataset (`data/valid.csv`):

| Metric | Score |
| :--- | :--- |
| **Baseline F1-Score** | `0.7958` |
| **Our Validation F1-Score** | **89.94%** |
| **Validation Accuracy** | **89.91%** |
| **Validation Precision** | **89.87%** |
| **Validation Recall** | **90.01%** |

---

## 🤗 Hugging Face Model Hub

The trained model has been uploaded publicly to Hugging Face:
* **Hugging Face Model Link:** `https://huggingface.co/egencer9/xlm-roberta-base-fine-tuned`

To pull and use the model directly in Python:
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="egencer9/xlm-roberta-base-fine-tuned")
print(classifier("Super vendeur !"))
```

---

## 🏃 How to Run the Evaluation Script

Ensure all dependencies are installed:
```bash
pip install torch transformers datasets evaluate scikit-learn pandas
```

To run the validation script:
```bash
python run_test.py --valid
```
