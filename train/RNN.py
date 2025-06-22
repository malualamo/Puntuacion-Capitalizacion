import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PunctuationCapitalizationRNN(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_punct_classes, num_cap_classes, dropout=0.3):
        super().__init__()
        self.bert = bert_model                        # ahora el modelo completo
        self.projection = nn.Linear(
            self.bert.config.hidden_size, hidden_dim
        )
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=2,
                           batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.punct_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_punct_classes)
        )
        self.cap_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cap_classes)
        )

    def forward(self, input_ids, attention_mask=None):
        # BERT retorna last_hidden_state: (B, T, H_bert)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        projected = self.projection(hidden_states)   # (B, T, hidden_dim)

        # mismo packing/padding de antes
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)
            packed = pack_padded_sequence(projected, lengths.cpu(),
                                          batch_first=True, enforce_sorted=False)
            rnn_out_packed, _ = self.rnn(packed)
            rnn_out, _ = pad_packed_sequence(
                rnn_out_packed, batch_first=True, total_length=projected.size(1)
            )
        else:
            rnn_out, _ = self.rnn(projected)

        rnn_out = self.dropout(rnn_out)
        punct_logits = self.punct_classifier(rnn_out)
        cap_logits   = self.cap_classifier(rnn_out)
        return punct_logits, cap_logits