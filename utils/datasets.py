import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Iterable, Callable
from transformers import PreTrainedTokenizerFast
import torch

class TextDataset(Dataset):

    def __init__(self, texts: Iterable[str],
                        labels: Iterable[int],
                        tokenizer: PreTrainedTokenizerFast,
                        max_seq_len: int):
        
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    @classmethod
    def from_df(cls, df: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, max_seq_len: int = 512, text_column: str = "text", target_column: str = "label"):

        texts = df[text_column].values
        labels = df[target_column].values

        return cls(texts, labels, tokenizer, max_seq_len)

    def __len__(self):
        
        return len(self.labels) 

    def __getitem__(self, i):

        tokens = self.tokenizer.encode_plus(
            self.texts[i],
            max_length = self.max_seq_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt"
        )

        label = self.labels[i]

        return tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze(), torch.tensor(label, dtype=torch.long)