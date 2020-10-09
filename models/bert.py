from .base_transformer_model import *
from transformers import BertForSequenceClassification

class BertForClassification(BaseTransformerModel):

    def __init__(self, bert_model: BertForSequenceClassification, 
                       metrics: Dict[str, Callable[[Real, Prediction], float]]):

        super().__init__(bert_model, metrics)

    def forward(self, input_ids, att_masks):
        # Use only the logits
        # TODO: Usar apenas BertModel e por uma camada linear na frente
        return self.transformer_encoder(input_ids, att_masks)[0]
