from argparse import ArgumentParser
import pandas as pd
import os
from transformers import TextClassificationPipeline, BertTokenizerFast, BertForSequenceClassification

if __name__ == "__main__":
    
    parser = ArgumentParser()

    parser.add_argument("--source_file", type=str, required=True, help="Arquivo .csv de entrada. O arquivo será sobrescrito com a adição de uma coluna com predições.")
    parser.add_argument("--text_column", type=str, required=True, help="Coluna de texto a ser processada do arquivo de entrada.")
    parser.add_argument("--bs", type=int, required=True, help="Tamanho do batch para inferência.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint para inferência.")

    parser.add_argument("--device", type=int, default=0, help="Device para inferência.")
    parser.add_argument("--tokenizer", type=str, default="neuralmind/bert-base-portuguese-cased", help="Tokenizer para utilização.")

    args = parser.parse_args()

    print(f"Loading data from {args.source_file}")
    data_df = pd.read_csv(args.source_file, index_col=0)
    print(data_df.tail())

    # Carregando checkpoint e tokenizador
    print(f"- Loading tokenizer {args.tokenizer}.")
    print(f"- Loading checkpoint {args.checkpoint}.")
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
    model = BertForSequenceClassification.from_pretrained(args.checkpoint)

    # Preparando pipeline de inferência
    print(f"- Preparing inference pipeline.")
    pipeline = TextClassificationPipeline(
        model=model, 
        tokenizer=tokenizer, 
        # batch_size=args.bs, # TODO: não funciona com batch_size, atualizar a lib transformers eventualmente.
        device=args.device
    )

    print(f"- Running inference:")
    preds_dict = pipeline(data_df[args.text_column].values.tolist())
    # Extraindo predições do dicionario retornado
    preds = [example["label"] for example in preds_dict]
    data_df["predictions"] = preds

    # Salvando resultados
    print(f"- Saving results to {args.source_file}.")
    data_df.to_csv(f"{args.source_file}")




    