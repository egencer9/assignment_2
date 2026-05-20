# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is the file for the model class.
# !     You need to implement the train and predict methods.
# !     You can also add any other methods as required.
# !     You can also add required parameters to the methods.
# !     You can also use additional packages in this file.
# !
# ! Make sure that the final implementation is compatible with the
# !     Model class. Be careful about the input and output types of
# !     the methods.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from datasets import DatasetDict
from typing import List, Dict
import numpy as np
import torch


def get_device():
    """
    Dynamically detects and returns the best available hardware accelerator.
    Supports TPU (Google Colab/Kaggle via PyTorch XLA), CUDA (NVIDIA GPU), 
    MPS (Apple Silicon GPU), and fallback to CPU.
    """
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except ImportError:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Model:
    def __init__(self, pretrained_path: str = None, **kwargs):
        self.pretrained_path = pretrained_path
        self.__dict__.update(kwargs)

        if pretrained_path != None:
            self.__dict__.update(self.load(self.pretrained_path).__dict__)
        else:
            self.init_model(kwargs)

    def init_model(self, *args, **kwargs):
        config_dict = {}
        if args and isinstance(args[0], dict):
            config_dict = args[0]
        config_dict.update(kwargs)

        self.pretrained_model_name = config_dict.get("pretrained_model_name", "xlm-roberta-base")
        self.num_labels = config_dict.get("num_labels", 2)

        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=self.num_labels
        )
        
        self.model.to(get_device())

    def train(self, datasets: List, training_args: Dict = None):
        from transformers import Trainer, TrainingArguments

        train_dataset = datasets[0]
        eval_dataset = datasets[1] if len(datasets) > 1 else None

        self.model.to(get_device())

        if training_args is None:
            training_args = {}

        num_epochs = training_args.get("num_train_epochs", 2)
        learning_rate = training_args.get("learning_rate", 2e-5)
        per_device_train_batch_size = training_args.get("per_device_train_batch_size", 16)
        per_device_eval_batch_size = training_args.get("per_device_eval_batch_size", 16)
        output_dir = training_args.get("output_dir", "./results")

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            weight_decay=0.01,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1" if eval_dataset else None,
            greater_is_better=True,
            logging_steps=100,
            report_to="none"
        )

        def compute_metrics_fn(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            metrics = self.compute_metrics(labels.tolist(), predictions.tolist())
            return metrics

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_fn if eval_dataset else None,
        )

        trainer.train()
        self.model.eval()

    def predict(self, x) -> List[int]:
        device = next(self.model.parameters()).device
        self.model.eval()

        # 1. Check if HF Dataset
        from datasets import Dataset
        if isinstance(x, Dataset):
            from transformers import Trainer
            trainer = Trainer(model=self.model)
            output = trainer.predict(x)
            predictions = np.argmax(output.predictions, axis=-1).tolist()
            return predictions

        # 2. Check if list of strings
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], str):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
            # Batch string predictions to prevent GPU Out Of Memory
            batch_size = 64
            predictions = []
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size]
                inputs = tokenizer(batch_x, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                predictions.extend(batch_preds)
            return predictions

        # 3. Check if HF dataset style dict
        elif isinstance(x, dict):
            inputs = {k: torch.tensor(v).to(device) if not isinstance(v, torch.Tensor) else v.to(device) for k, v in x.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            return predictions

        # 4. Check if list of list of ints (input IDs)
        elif isinstance(x, list) and len(x) > 0 and isinstance(x[0], list) and isinstance(x[0][0], int):
            input_ids = torch.tensor(x).to(device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            return predictions

        return []

    def compute_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict:
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        return {
            "f1": float(f1_score(y_true, y_pred, average="binary")),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="binary")),
            "recall": float(recall_score(y_true, y_pred, average="binary"))
        }

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "model" in state:
            # Move model to CPU before serialization to prevent CUDA/MPS/TPU loading failures on other platforms
            model_cpu = self.model.cpu()
            state["model_state_dict"] = model_cpu.state_dict()
            del state["model"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=self.num_labels
        )
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
            del state["model_state_dict"]

        self.model.to(get_device())
