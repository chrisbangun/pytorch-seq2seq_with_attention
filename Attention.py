import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size, query_size, memory_size):
        super(BahdanauAttention, self).__init__()

        self.hidden_size = hidden_size
        self.sofmax = nn.Softmax()

        self.query_layer = nn.Linear(query_size, hidden_size, bias=False) 
        self.memory_layer = nn.Linear(memory_size, hidden_size, bias=False) 
        self.alignment_layer = nn.Linear(hidden_size, 1, bias=False)

    def alignment_score(self, query, keys):
        query = self.query_layer(query)
        keys = self.memory_layer(keys)

        extendded_query = query.unsqueeze(1)
        alignment = self.alignment_layer(F.tanh(extendded_query + keys))

        return alignment.squezee(2)

    def forward(self, query, keys):
        alignment_score = self.alignment_score(query, keys)
        weight = F.softmax(alignment_score)

        context = weight.unsqueeze(2) * keys

        total_context = context.sum(1)
        
        return total_context