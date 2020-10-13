from .base_transformer_model import *
from transformers import BertModel

class BertForClassification(BaseTransformerModel):

    def __init__(self, bert_model: BertModel,
                       classes: int,
                       metrics: Dict[str, Callable[[Real, Prediction], float]],
                       custom_head: nn.Sequential = None):

        super().__init__(bert_model, metrics)

        if custom_head is not None:
            self.classifier = custom_head
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(bert_model.config.hidden_dropout_prob),
                nn.Linear(bert_model.config.hidden_size, classes)
            )

    def forward(self, input_ids, att_masks):
        
        outputs = self.transformer_encoder(input_ids, att_masks)
        pooled_outputs = outputs[1]

        return self.classifier(pooled_outputs)
