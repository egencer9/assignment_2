import os
import pickle
import pandas as pd
import torch
from datasets import Dataset
from preprocessing import Preprocessor, Tokenizer, Embedder
from model import Model

def main():
    print("=" * 60)
    print("CMPE 346 Multilingual Sentiment Analysis Training Pipeline")
    print("=" * 60)

    # 1. Check GPU Device
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}")

    # 2. Load Datasets
    print("\n[1/6] Loading CSV datasets...")
    train_df = pd.read_csv("data/train.csv")
    valid_df = pd.read_csv("data/valid.csv")
    print(f"Loaded {len(train_df)} training samples and {len(valid_df)} validation samples.")

    # 3. Preprocess Datasets
    print("\n[2/6] Initializing Preprocessor and Tokenizer...")
    # Using 'xlm-roberta-base' as the pre-trained multilingual foundation
    pretrained_model = "xlm-roberta-base"
    tokenizer = Tokenizer(pretrained_model_name=pretrained_model)
    preprocessor = Preprocessor(tokenizer)

    train_df = preprocessor.prepare_data(train_df)
    valid_df = preprocessor.prepare_data(valid_df)

    # Convert textual labels ('positive', 'negative') to integer IDs (1, 0)
    train_df['label_id'] = train_df['label'].map({"negative": 0, "positive": 1})
    valid_df['label_id'] = valid_df['label'].map({"negative": 0, "positive": 1})

    # 4. Stratified Subset Selection for Efficient GPU Training
    # Fine-tuning a 270M parameter Transformer on 140k samples can take hours.
    # A stratified sample of 20k balanced rows gives excellent generalization and trains in ~15 mins.
    print("Creating balanced training subset...")
    subset_size_per_class = 10000  # Total 20,000 samples
    train_df_subset = train_df.groupby('label_id', group_keys=False).apply(
        lambda x: x.sample(n=subset_size_per_class, random_state=42)
    )
    print(f"Training subset size: {len(train_df_subset)} (perfectly balanced)")

    # 5. Tokenize and Build HF Datasets
    print("\n[3/6] Tokenizing datasets...")
    # Map function for batch tokenization
    def tokenize_fn(examples):
        # We use a max_length of 128 for training efficiency. It captures the essential sentiment
        # of review texts perfectly while accelerating training 4x and using much less memory.
        return tokenizer.tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    train_dataset = Dataset.from_pandas(train_df_subset[['text', 'label_id']])
    valid_dataset = Dataset.from_pandas(valid_df[['text', 'label_id']])

    # Stratified validation subset for fast evaluation during training epochs
    valid_df_subset = valid_df.groupby('label_id', group_keys=False).apply(
        lambda x: x.sample(n=1000, random_state=42)
    )
    valid_dataset_eval = Dataset.from_pandas(valid_df_subset[['text', 'label_id']])

    train_dataset = train_dataset.rename_column("label_id", "label")
    valid_dataset = valid_dataset.rename_column("label_id", "label")
    valid_dataset_eval = valid_dataset_eval.rename_column("label_id", "label")

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    valid_dataset = valid_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    valid_dataset_eval = valid_dataset_eval.map(tokenize_fn, batched=True, remove_columns=['text'])

    # 6. Initialize Model and Fine-tune
    print("\n[4/6] Initializing Multilingual Transformer Model...")
    model = Model(pretrained_model_name=pretrained_model, num_labels=2)

    print("\n[5/6] Starting model fine-tuning...")
    # Define training arguments for our model
    training_params = {
        "num_train_epochs": 2,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 32,
        "output_dir": "./results"
    }

    # Run the training loop
    model.train([train_dataset, valid_dataset_eval], training_args=training_params)

    # 7. Local Evaluation and Verification
    print("\n[6/6] Running validation set evaluation...")
    eval_predictions = model.predict(valid_dataset)
    ground_truth = valid_dataset['label']
    metrics = model.compute_metrics(ground_truth, eval_predictions)
    print(f"\nFinal Validation Metrics:")
    for metric_name, val in metrics.items():
        print(f"  - {metric_name.upper()}: {val:.4f}")

    # 8. Save Model, Tokenizer, and Embedder pickle files
    print("\nSaving files to saved_objects/...")
    os.makedirs("saved_objects", exist_ok=True)

    # A. Save model.pkl (using Model's custom CPU-safe state_dict serialization)
    print("Saving model.pkl...")
    model.save("saved_objects/model.pkl")

    # B. Save tokenizer.pkl (using custom pickle states to avoid C++ binding issues)
    print("Saving tokenizer.pkl...")
    with open("saved_objects/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # C. Save embedder.pkl
    print("Saving embedder.pkl...")
    embedder = Embedder(name="xlm-roberta-base-embeddings")
    with open("saved_objects/embedder.pkl", "wb") as f:
        pickle.dump(embedder, f)

    # D. Also save HF model weights/tokenizer files locally so the user can easily push them to the Hub
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
    print("\nTo upload your model to Hugging Face Hub, run:")
    print("  huggingface-cli login")
    print("  python3 -c \"from transformers import AutoModelForSequenceClassification, AutoTokenizer; \\")
    print("  m = AutoModelForSequenceClassification.from_pretrained('saved_model'); \\")
    print("  t = AutoTokenizer.from_pretrained('saved_model'); \\")
    print("  m.push_to_hub('your-username/your-repo-name'); \\")
    print("  t.push_to_hub('your-username/your-repo-name')\"")
    print("=" * 60)

if __name__ == "__main__":
    main()
