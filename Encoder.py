import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, input):
        """
        argument:

        """
        embedded = self.embedding(input).view(1, 1, -1)
        output, _ = self.gru(embedded)
        return output
