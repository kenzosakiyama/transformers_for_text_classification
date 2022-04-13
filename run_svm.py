from argparse import ArgumentParser
from functools import partial
from typing import Iterable, Tuple 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

import spacy

import numpy as np
import math


SVM_PARAMS = {
   "kernel": "linear", 
   "class_weight": "balanced",
   "random_state": 1234,
   "verbose": True
}

TFIDF_PARAMS = {
    "ngram_range": (1,2),
    "max_features": 70000,
    "min_df": 2,
    "max_df": 74608, # metade do número de exemplos do conjunto de treino.
    "tokenizer": None
}

def load_train_test_data(train_path: str,
                         test_path: str,
                         text_col: str,
                         label_col: str) -> Tuple[LabelEncoder, pd.DataFrame, pd.DataFrame]:

    # Carregando DataFrames e eliminando colunas vazias 
    train_df = pd.read_csv(train_path, index_col=0).dropna()
    test_df = pd.read_csv(test_path, index_col=0).dropna()
    # Filtrando colunas de interesse (texto e classe)
    train_df = train_df[[text_col, label_col]]
    test_df = test_df[[text_col, label_col]]

    # Para teste
    # train_df = train_df.sample(100)
    # test_df = train_df.copy()
    # test_df = test_df.sample(100)

    label_encoder = LabelEncoder()

    train_df[label_col] = label_encoder.fit_transform(train_df[label_col])
    test_df[label_col] = label_encoder.transform(test_df[label_col])

    return (label_encoder, train_df, test_df)

def apply_preprocessing_pipeline(text: str, use_lemmatization: bool = True, remove_stopwords: bool = True, nlp: spacy.lang = None) -> Iterable[str]:

    if nlp is None: nlp = spacy.load("pt_core_news_md") # Carrega modelo do spaCy básico
    if len(text) > nlp.max_length: text = text[:nlp.max_length]

    tokens = nlp(text)
    final_tokens = []

    for token in tokens:

        lemma = token.lemma_
        if token.is_stop and remove_stopwords: continue # remoção de stop-words
        if token.is_punct: continue             # remoção de pontuação
        if len(lemma.strip()) == 0: continue    # ignorar tokens "vazios"
        # if token.ent_type_ == "PER": continue   # remoção de nomes próprios usando NER
        if token.pos_ == "NUM": continue        # remoção de números e datas usando POS tag
        if lemma.count(".") > 1: continue       # remoção de datas no formato DD.MM.AA
        # if "unâni" in lemma.lower() or "unani" in lemma.lower(): continue   # remoção de palavras frequentes como "unanime" e "unanimamente" 

        # Permitindo usar ou não lemmatização.
        if use_lemmatization: final_tokens.append(lemma.lower())
        else:                 final_tokens.append(token.text.lower())
    
    return final_tokens

def load_vectorizer(nlp, remove_stopwords: bool, use_lemmatization: bool):

    global TFIDF_PARAMS

    print(f"Loading vectorizer using:")
    print(f"\t- Use lemmatization: {use_lemmatization}")
    print(f"\t- Remove stop words: {remove_stopwords}")
    print(f"\t- Remaining args: {TFIDF_PARAMS}")

    tok_func = partial(apply_preprocessing_pipeline, 
                       nlp=nlp,
                       use_lemmatization=use_lemmatization,
                       remove_stopwords=remove_stopwords)

    TFIDF_PARAMS["tokenizer"] = tok_func

    return TfidfVectorizer(**TFIDF_PARAMS)

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--train_file", type=str, required=True, help="Path to the train CSV file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test CSV file.")

    parser.add_argument("--remove_stopwords", action="store_true", help="Wether remove or not the stopwords from the text.")
    parser.add_argument("--use_lemma", action="store_true", help="Wether use or not lemmatization.")
    parser.add_argument("--text_col", type=str, default="text", help="Column containing the text examples.")
    parser.add_argument("--label_col", type=str, default="label", help="Column containing the labels.")
    parser.add_argument("--save_test_preds", type=str, default=None, help="Path to save test predictions.")

    args = parser.parse_args()

    print(f"- Loading train/test files")
    label_encoder, train_df, test_df = load_train_test_data(
        args.train_file,
        args.test_file,
        args.text_col,
        args.label_col
    )

    print(f"\t- Train Dataset: {train_df.info()}")
    print(f"\t- Test Dataset: {test_df.info()}")

    print(f"- Loading model and vectorizer:")

    nlp = spacy.load("pt_core_news_md")
    vectorizer = load_vectorizer(nlp, args.remove_stopwords, args.use_lemma)
    svc = SVC(**SVM_PARAMS)

    train_x = vectorizer.fit_transform(train_df[args.text_col])
    train_y = train_df[args.label_col]

    test_x = vectorizer.transform(test_df[args.text_col])
    test_y = test_df[args.label_col]

    print(f"- Training SVM: {svc}")
    svc.fit(train_x, train_y)

    print(f"- Evaluating:")
    y_pred = svc.predict(test_x)
    print(classification_report(test_y, y_pred))

    if args.save_test_preds:
        print(f"- Saving predictions to {args.save_test_preds}")
        test_df["prediction"] = label_encoder.inverse_transform(y_pred)
        test_df[args.label_col] = label_encoder.inverse_transform(test_df[args.label_col])
        test_df.to_csv(args.save_test_preds)


