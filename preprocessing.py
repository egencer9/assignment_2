from datasets import Dataset
import pandas as pd
from typing import List, Dict


class Preprocessor:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.label_encoder = LabelEncoder(labels={"negative": 0, "positive": 1})
        self.__dict__.update(kwargs)

    def prepare_data(self, dataset) -> pd.DataFrame:
        # run_test.py buraya pandas DataFrame gönderiyor.
        # Eğer veri üzerinde ekstra bir temizleme (lowercase, noktalama temizliği vb.) 
        # yapmak istersen burada yapabilirsin. Çok dilli modeller için ham metin genelde yeterlidir.
        if isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
        else:
            df = pd.DataFrame(dataset)

        if "text" in df.columns:
            # Clean whitespaces and fill NaNs
            df["text"] = df["text"].astype(str).str.strip()

        return df


class LabelEncoder:
    def __init__(self, labels, **kwargs):
        self.id2label = {v: k for k, v in labels.items()}
        self.label2id = {k: v for k, v in labels.items()}
        self.__dict__.update(kwargs)


class Tokenizer:
    def __init__(self, pretrained_model_name: str = "xlm-roberta-base", **kwargs):
        from transformers import AutoTokenizer
        self.pretrained_model_name = pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.__dict__.update(kwargs)

    def train(self, texts: List[str]):
        # Using pre-trained multilingual vocab; training from scratch is not required.
        pass

    def tokenize(self, text: str) -> Dict:
        return self.tokenizer(text, truncation=True, max_length=512)

    def push_to_hub(self, path: str):
        self.tokenizer.push_to_hub(path)

    def from_pretrained(self, path: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        return self

    def __getstate__(self):
        # Bulletproof pickling: exclude local tokenizers containing non-picklable C++ bindings,
        # save the pretrained name to reload seamlessly upon unpickling.
        state = self.__dict__.copy()
        if "tokenizer" in state:
            del state["tokenizer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)


class Embedder:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)