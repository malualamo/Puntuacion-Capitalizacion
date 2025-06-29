import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PunctuationCapitalizationRNNBidirectionalAttention(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_punct_classes, num_cap_classes, dropout=0.3):
        super().__init__()
        self.bert = bert_model
        bert_dim = bert_model.config.hidden_size

        self.projection = nn.Linear(bert_dim, hidden_dim)

        self.lstm1 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.punct_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_punct_classes),
        )
        self.cap_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cap_classes),
        )

    def forward(self, input_ids, attention_mask=None):
            bert_out = self.bert(input_ids, attention_mask=attention_mask)
            x = self.projection(bert_out.last_hidden_state)       # [B, T, hidden_dim]

            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).cpu()
                packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                packed1, _ = self.lstm1(packed)
                x, _ = pad_packed_sequence(packed1, batch_first=True,
                                        total_length=attention_mask.size(1))
            else:
                x, _ = self.lstm1(x)

            if attention_mask is not None:
                key_padding_mask = attention_mask == 0
            else:
                key_padding_mask = None

            attn_out, _ = self.attention(
                x, x, x,
                key_padding_mask=key_padding_mask
            )

            if attention_mask is not None:
                packed2 = pack_padded_sequence(attn_out, lengths, batch_first=True, enforce_sorted=False)
                packed3, _ = self.lstm2(packed2)
                x, _ = pad_packed_sequence(packed3, batch_first=True,
                                        total_length=attention_mask.size(1))
            else:
                x, _ = self.lstm2(attn_out)

            x = self.dropout(x)

            punct_logits = self.punct_classifier(x)
            cap_logits   = self.cap_classifier(x)
            return punct_logits, cap_logits
