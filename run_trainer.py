from argparse import ArgumentParser
from typing import Tuple 
from datasets import Dataset, load_metric
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import math

from transformers import DataCollatorWithPadding, EarlyStoppingCallback
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments, get_cosine_schedule_with_warmup
from torch.optim import AdamW

TOKENIZER_BS = 5000
MAX_SEQ_LEN = 512

def load_train_test_data(train_path: str,
                         test_path: str,
                         text_col: str,
                         label_col: str) -> Tuple[LabelEncoder, Dataset, Dataset]:

    # Carregando DataFrames e eliminando colunas vazias 
    train_df = pd.read_csv(train_path, index_col=0).dropna()
    test_df = pd.read_csv(test_path, index_col=0).dropna()
    # Filtrando colunas de interesse (texto e classe)
    train_df = train_df[[text_col, label_col]]
    test_df = test_df[[text_col, label_col]]

    # Para teste
    # train_df = train_df.sample(100)
    # test_df = test_df.sample(100)

    label_encoder = LabelEncoder()

    train_df[label_col] = label_encoder.fit_transform(train_df[label_col])
    test_df[label_col] = label_encoder.transform(test_df[label_col])

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    return (label_encoder, train_ds, test_ds)

def tokenize_inputs(ds: Dataset, 
                    text_col: str,
                    tokenizer: BertTokenizerFast,
                    bs: int = TOKENIZER_BS,
                    max_seq_len: int = MAX_SEQ_LEN) -> Dataset:

    def tokenizer_function(example):
        # Em caso de uma tarefa de classificação de pares de texto, modificar este valor de retorno
        # truncation=True, padding="max_length", max_length=123 para truncar e padronizar os tamanhos de tokens!!!
        return tokenizer(
            example[text_col], truncation=True, max_length=max_seq_len
        )

    # Tokenizando todos os elementos do conjunto de dados em batches
    ds = ds.map(tokenizer_function, batched=True, batch_size=bs)

    return ds

def prepare_inputs(ds: Dataset, 
                   text_col: str,
                   label_col: str) -> Dataset:

    ds = ds.remove_columns(column_names=[text_col, "__index_level_0__"])
    ds = ds.rename_column(label_col, "labels")
    ds = ds.with_format("torch")

    return ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics_dict = classification_report(labels, predictions, output_dict=True)

    return {
        "accuracy": metrics_dict["accuracy"],
        "precision": metrics_dict["macro avg"]["precision"],
        "recall": metrics_dict["macro avg"]["recall"],
        "f1": metrics_dict["macro avg"]["f1-score"]
    }

def get_collator(tokenizer: BertTokenizerFast,
                 max_seq_len: int = MAX_SEQ_LEN):
    collator = DataCollatorWithPadding(tokenizer, padding="longest", max_length=max_seq_len)

    return collator

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--train_file", type=str, required=True, help="Path to the train CSV file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test CSV file.")

    parser.add_argument("--model_name", type=str, default="label", help="Column containing the labels.")
    parser.add_argument("--epochs", type=int, help="Number of epochs of training.")
    parser.add_argument("--bs", type=int, help="Batch size for training.")
    parser.add_argument("--lr", type=float, help="Learning rate for training.")
    parser.add_argument("--checkpoint_dir", type=str, help="Path to store the model checkpoints.")

    parser.add_argument("--patience", type=int, default=5, help="Number of epochs of patience. Used for early-stopping.")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--text_col", type=str, default="text", help="Column containing the text examples.")
    parser.add_argument("--label_col", type=str, default="label", help="Column containing the labels.")

    args = parser.parse_args()

    print(f"- Loading train/test files")
    label_encoder, train_ds, test_ds = load_train_test_data(
        args.train_file,
        args.test_file,
        args.text_col,
        args.label_col
    )

    print(f"\t- Train Dataset: {train_ds}")
    print(f"\t- Test Dataset: {test_ds}")

    print(f"- Loading model and tokenizer ({args.model_name})")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=len(label_encoder.classes_))

    print(f"\t- Tokenizing data.")
    train_ds = tokenize_inputs(train_ds, args.text_col, tokenizer)
    test_ds = tokenize_inputs(test_ds, args.text_col, tokenizer)
    print(f"\t- Preparing inputs for training and evaluation.")
    train_ds = prepare_inputs(train_ds, args.text_col, args.label_col)
    test_ds = prepare_inputs(test_ds, args.text_col, args.label_col)

    warmup_steps = math.ceil((len(train_ds)/args.bs) * args.epochs * 0.1) #10% of train data for warm-up
    train_steps = int(args.epochs * len(train_ds)/args.bs)

    optimizer = AdamW(model.parameters(), lr=args.lr) 
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps, num_cycles=0.5)

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,                                                                                # output directory
        num_train_epochs=args.epochs,                                                                              # total number of training epochs
        per_device_train_batch_size=args.bs,                                                                       # batch size per device during training
        per_device_eval_batch_size=args.bs,                                                                        # batch size for evaluation
        # warmup_steps=warmup_steps,                                                                            # number of warmup steps for learning rate scheduler
        weight_decay=args.wd,                                                                                   # strength of weight decay
        evaluation_strategy="epoch",                                                                          # evaluation interval
        logging_dir=args.checkpoint_dir,                                                                                 # directory for storing logs
        save_strategy="epoch",                                                                                # checkpoint save interval
        logging_steps=500,
        metric_for_best_model="f1",
        load_best_model_at_end=True
    )

    collator = get_collator(tokenizer)
    es_callback = EarlyStoppingCallback(early_stopping_patience=5)

    print(f"- Training args: {training_args}")
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        data_collator=collator
    )

    trainer.add_callback(es_callback)

    trainer.train()