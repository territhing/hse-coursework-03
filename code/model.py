import torch
from torch import nn
from transformers import BertForPreTraining


class BertRestorePunctuation(nn.Module):
    def __init__(self, bert: BertForPreTraining, num_classes, dropout_prob=0.1, hidden_size=512):
        super().__init__()
        self.bert = bert
        self.embedding_size = self.bert.config.hidden_size
        self.clf = nn.Sequential(
            nn.Linear(self.embedding_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.bert(x, attention_mask=attention_mask)["hidden_states"][0]
        return self.clf(embeddings)
