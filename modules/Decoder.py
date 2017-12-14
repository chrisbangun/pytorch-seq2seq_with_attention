import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(self, hidden_size, embedding_size, output_size, n_layers):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding  = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()


    def forward(self, input):
        output = self.embedding(input).view(1, 1, -1)
        output, _ = self.gru(output)
        output = self.softmax(self.out(output))
        return output

