from .base_transformer_model import *
from transformers import GPT2Model

class GPT2ForClassification(BaseTransformerModel):

    def __init__(self, gpt2_model: GPT2Model, 
                       max_seq_len: int,
                       classes: int,
                       metrics: Dict[str, Callable[[Real, Prediction], float]],
                       dropout: float = 0.1,
                       device: Union[str, torch.device] = "cpu"):

        super().__init__(gpt2_model, metrics, device=device)
        concat_features = gpt2_model.config.n_embd * max_seq_len
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(concat_features, classes)


    def concatenate_hidden_states(self, hidden_states: torch.Tensor, bs: int):

        return hidden_states.view(bs, -1)


    def forward(self, input_ids, att_masks):
        
        hidden_states = self.transformer_encoder(input_ids=input_ids, attention_mask=att_masks)[0]
        batch_size = hidden_states.shape[0]
        # É possível tentar outras abordagens para sumarizar todos os hidden states
        features = self.concatenate_hidden_states(hidden_states, batch_size)
        # print(features.shape) Descobrindo dimensões no gambito
        output = self.classifier(self.dropout(features))

        return output 