from .base_transformer_model import *
from transformers import BertModel

class BertForClassification(BaseTransformerModel):

    def __init__(self, bert_model: BertModel,
                       classes: int,
                       metrics: Dict[str, Callable[[Real, Prediction], float]],
                       custom_head: nn.Sequential = None,
                       device: Union[str, torch.device] = "cpu"):

        super().__init__(bert_model, metrics, device=device)

        self.max_seq_len = 128 # TODO: remover em caso de resultado positivo
        concat_features = bert_model.config.hidden_size * self.max_seq_len

        if custom_head is not None:
            self.classifier = custom_head
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(concat_features, classes)
            )

    # Original
    # def forward(self, input_ids, att_masks):
        
    #     outputs = self.transformer_encoder(input_ids, att_masks)
    #     pooled_outputs = outputs[1]

    #     return self.classifier(pooled_outputs)

    def concatenate_hidden_states(self, hidden_states: torch.Tensor, bs: int):

        return hidden_states.view(bs, -1)

    def forward(self, input_ids, att_masks):
        
        outputs = self.transformer_encoder(input_ids, att_masks)
        hidden_states = outputs[0]
        batch_size = hidden_states.shape[0]

        features = self.concatenate_hidden_states(hidden_states, batch_size)

        return self.classifier(features)

