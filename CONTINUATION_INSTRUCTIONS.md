# CMPE 346 - Sentiment Analysis Continuation Instructions

This document provides clear, structured instructions so that another AI model or developer can seamlessly pick up where we left off and complete the training, evaluation, and packaging of this project.

---

## 📍 Current Project Status
All core classes, pipelines, configurations, and evaluation helpers are fully implemented and optimized:
- **`preprocessing.py`**: Fully implemented with `Preprocessor`, robust `Tokenizer` (with C++ binding pickling safety hooks), and `Embedder` wrapper classes.
- **`model.py`**: Fully implemented with custom `get_device()` to support out-of-the-box training on Google Colab GPUs (CUDA) and TPUs (XLA), Apple Silicon (MPS), and CPUs.
- **`main.py`**: A fully configurable training coordinator that reads hyperparameters directly from `config.json`.
- **`config.json`**: Outlines model checkpoints, training hyperparameters, sequence padding limits, and subset sizes.
- **`run_test.py`**: Evaluation script (starter code bug fixed: `from preprocessing import Preprocessor` added).
- **`.gitignore`**: Excludes the large datasets and temporary cache directories while keeping code clean.

The repository code is committed and pushed to:
👉 **[https://github.com/egencer9/assignment_2](https://github.com/egencer9/assignment_2)**

---

## 🛠️ Step-by-Step Training Guide (Google Colab GPU / TPU)

Follow these instructions to run the training on Google Colab (either using a T4/A100 GPU or TPU v2/v3):

### Step 1: Open Google Colab and Clone the Repository
Create a new Google Colab notebook, change the runtime type to **GPU** (T4/A100) or **TPU**, and execute:
```bash
# Clone the clean code repository
!git clone https://github.com/egencer9/assignment_2.git
%cd assignment_2
```

### Step 2: Install Required Libraries
Install the deep learning and transformer libraries. 
* **If using GPU (T4/A100):**
  ```bash
  !pip install torch transformers datasets evaluate scikit-learn pandas tqdm
  ```
* **If using TPU:**
  Install PyTorch XLA compatibility libraries:
  ```bash
  !pip install torch-xla transformers datasets evaluate scikit-learn pandas tqdm
  ```

### Step 3: Upload the Large Data Files
Because the large datasets `train.csv` (51.8 MB) and `valid.csv` (6.3 MB) are excluded from git, you need to upload them to the Colab instance:
1. Create a `data/` directory:
   ```bash
   !mkdir -p data
   ```
2. Upload the two files (`train.csv` and `valid.csv`) into the `data/` folder (using the file manager sidebar in Colab or mounting Google Drive).

### Step 4: Run the Training Pipeline
Run `main.py`. This script will:
1. Dynamically read configurations from `config.json`.
2. Select the optimal hardware accelerator (TPU/GPU/CPU).
3. Preprocess and tokenize text data using sequence truncation length `128` (giving 4x training speedups).
4. Perform stratified subset selection (20,000 perfectly balanced training rows).
5. Fine-tune `xlm-roberta-base` for 2 epochs on GPU/TPU.
6. Evaluate on the full 17,500 validation rows.
7. Save serialized compact `tokenizer.pkl`, `embedder.pkl`, and `model.pkl` to `saved_objects/` (with cross-platform loading safety guarantees).
8. Save raw model weights to `saved_model/`.

Execute:
```bash
!python main.py
```

### Step 5: Verify via `run_test.py`
Before publishing, test locally using the instructor's validation script:
1. Update `model_checkpoint = "saved_model"` in `run_test.py` around line 67 to test loading from your locally saved weights.
2. Execute the verification script:
   ```bash
   !python run_test.py --valid
   ```
Ensure that the printed F1-Score surpasses the **0.7958** baseline comfortably.

### Step 6: Push Model to Hugging Face Hub
Register / log in to your Hugging Face account and push the fine-tuned model weights so they are publicly accessible:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Replace with your credentials
username = "egencer9"
repo_name = "xlm-roberta-sentiment-multilingual"
hf_repo = f"{username}/{repo_name}"

# Load from our training outputs
model = AutoModelForSequenceClassification.from_pretrained("saved_model")
tokenizer = AutoTokenizer.from_pretrained("saved_model")

# Push publicly to Hugging Face Model Hub
model.push_to_hub(hf_repo)
tokenizer.push_to_hub(hf_repo)
```

### Step 7: Final Updates & Packaging
1. **Update `run_test.py`:** Change `model_checkpoint = "huggingface-link"` on line 68 to your public HF identifier, e.g., `model_checkpoint = "egencer9/xlm-roberta-sentiment-multilingual"`.
2. **Update `README.md`:** Fill in the validation F1, accuracy, precision, and recall scores you obtained, and insert your Hugging Face model link.
3. **Commit & Push to Git:**
   ```bash
   !git add README.md run_test.py
   !git commit -m "Add final evaluation results and Hugging Face model link"
   !git push origin main
   ```
4. **Create Zip Archive for Submission:**
   Create the final submission ZIP file with the precise directory structure:
   ```bash
   !zip -r ege_gencer_assignment02.zip data/ main.py model.py preprocessing.py run_test.py README.md saved_objects/
   ```
   Download `ege_gencer_assignment02.zip` from your Colab files panel to submit it!

---

## 💡 Notes for the Continuing Model
* **Batch Size Tuning:** If you hit CUDA Out-Of-Memory (OOM) errors on low-memory GPUs, edit `config.json` to decrease `per_device_train_batch_size` to `16` or `8`.
* **Zero C++ Binding Issues:** The `Tokenizer` class has custom `__getstate__` and `__setstate__` hooks to prevent pickling issues with the tokenizer's underlying Rust/C++ fast tokenizers. Do not modify these.
* **CPU Serialization:** The `Model` class has custom serialization hooks that move model weights to CPU before pickling, preventing device loading failures during evaluation. Keep this intact.
