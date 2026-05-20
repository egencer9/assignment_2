import os
import json
import pickle
import pandas as pd
import torch
from datasets import Dataset
from preprocessing import Preprocessor, Tokenizer, Embedder
from model import Model, get_device

def main():
    print("=" * 60)
    print("CMPE 346 Multilingual Sentiment Analysis Training Pipeline")
    print("=" * 60)

    # 1. Load Configurations from config.json
    print("\nLoading training configuration from config.json...")
    if not os.path.exists("config.json"):
        raise FileNotFoundError("config.json not found in workspace!")
    with open("config.json", "r") as f:
        config = json.load(f)

    pretrained_model = config.get("pretrained_model_name", "xlm-roberta-base")
    num_labels = config.get("num_labels", 2)
    train_subset_size = config.get("train_subset_size_per_class", 10000)
    eval_subset_size = config.get("eval_subset_size_per_class", 1000)
    max_seq_length = config.get("max_length", 128)

    # 2. Check TPU/GPU/MPS Accelerator Device
    device = get_device()
    print(f"Using device for training: {device}")

    # 3. Load CSV Datasets
    print("\n[1/6] Loading CSV datasets...")
    train_df = pd.read_csv("data/train.csv")
    valid_df = pd.read_csv("data/valid.csv")
    print(f"Loaded {len(train_df)} training samples and {len(valid_df)} validation samples.")

    # 4. Preprocess Datasets
    print("\n[2/6] Initializing Preprocessor and Tokenizer...")
    tokenizer = Tokenizer(pretrained_model_name=pretrained_model)
    preprocessor = Preprocessor(tokenizer)

    train_df = preprocessor.prepare_data(train_df)
    valid_df = preprocessor.prepare_data(valid_df)

    # Convert textual labels ('positive', 'negative') to integer IDs (1, 0)
    train_df['label_id'] = train_df['label'].map({"negative": 0, "positive": 1})
    valid_df['label_id'] = valid_df['label'].map({"negative": 0, "positive": 1})

    # 5. Stratified Subset Selection for Efficient GPU/TPU Training
    print("Creating balanced training subset...")
    train_df_subset = train_df.groupby('label_id', group_keys=False).apply(
        lambda x: x.sample(n=train_subset_size, random_state=42)
    )
    print(f"Training subset size: {len(train_df_subset)} (perfectly balanced)")

    # 6. Tokenize and Build HF Datasets
    print("\n[3/6] Tokenizing datasets...")
    def tokenize_fn(examples):
        return tokenizer.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_seq_length)

    train_dataset = Dataset.from_pandas(train_df_subset[['text', 'label_id']])
    valid_dataset = Dataset.from_pandas(valid_df[['text', 'label_id']])

    # Stratified validation subset for fast evaluation during training epochs
    valid_df_subset = valid_df.groupby('label_id', group_keys=False).apply(
        lambda x: x.sample(n=eval_subset_size, random_state=42)
    )
    valid_dataset_eval = Dataset.from_pandas(valid_df_subset[['text', 'label_id']])

    train_dataset = train_dataset.rename_column("label_id", "label")
    valid_dataset = valid_dataset.rename_column("label_id", "label")
    valid_dataset_eval = valid_dataset_eval.rename_column("label_id", "label")

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    valid_dataset = valid_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    valid_dataset_eval = valid_dataset_eval.map(tokenize_fn, batched=True, remove_columns=['text'])

    # 7. Initialize Model and Fine-tune
    print("\n[4/6] Initializing Multilingual Transformer Model...")
    model = Model(pretrained_model_name=pretrained_model, num_labels=num_labels)

    print("\n[5/6] Starting model fine-tuning...")
    training_params = {
        "num_train_epochs": config.get("num_train_epochs", 2),
        "learning_rate": config.get("learning_rate", 2e-5),
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 32),
        "per_device_eval_batch_size": config.get("per_device_eval_batch_size", 32),
        "output_dir": config.get("output_dir", "./results")
    }

    # Run the training loop
    model.train([train_dataset, valid_dataset_eval], training_args=training_params)

    # 8. Local Evaluation and Verification
    print("\n[6/6] Running validation set evaluation...")
    eval_predictions = model.predict(valid_dataset)
    ground_truth = valid_dataset['label']
    metrics = model.compute_metrics(ground_truth, eval_predictions)
    print(f"\nFinal Validation Metrics:")
    for metric_name, val in metrics.items():
        print(f"  - {metric_name.upper()}: {val:.4f}")

    # 9. Save Model, Tokenizer, and Embedder pickle files
    print("\nSaving files to saved_objects/...")
    os.makedirs("saved_objects", exist_ok=True)

    print("Saving model.pkl...")
    model.save("saved_objects/model.pkl")

    print("Saving tokenizer.pkl...")
    with open("saved_objects/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("Saving embedder.pkl...")
    embedder = Embedder(name="xlm-roberta-base-embeddings")
    with open("saved_objects/embedder.pkl", "wb") as f:
        pickle.dump(embedder, f)

    print("\nSaving local model checkpoint for easy Hugging Face upload...")
    os.makedirs("saved_model", exist_ok=True)
    model.model.save_pretrained("saved_model")
    tokenizer.tokenizer.save_pretrained("saved_model")

    print("\n" + "=" * 60)
    print("TRAINING AND SERIALIZATION SUCCESSFULLY COMPLETED!")
    print("=" * 60)
    print("The following pickle files are ready under saved_objects/:")
    print("  - saved_objects/tokenizer.pkl")
    print("  - saved_objects/embedder.pkl")
    print("  - saved_objects/model.pkl")
    print("=" * 60)

if __name__ == "__main__":
    main()
