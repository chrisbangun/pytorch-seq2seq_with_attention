import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from Encoder import Encoder
from Decoder import Decoder
from Attention import BahdanauAttention


class Model(nn.Module):

    def __init__(self, vocab_source_size, vocab_target_size, embedding_size_encoder, embedding_size_decoder, hidden_size_encoder,
                 hidden_size_decoder, hidden_size_attention):

        self.hidden_size_attention = hidden_size_attention

        self.encoder = Encoder(vocab_source_size,
                          embedding_size_encoder, hidden_size_encoder)
        self.decoder = Decoder(vocab_target_size,
                          embedding_size_decoder, hidden_size_decoder)

        self.attention = BahdanauAttention(
            hidden_size_attention, hidden_size_decoder, hidden_size_encoder)
        
        
    def forward(self, source, target):

        output_encoder = self.encoder(source)
        batch_size = source.size(0)
        attention = torch.zeros(batch_size, self.hidden_size_attention)
        output_decoders = []
        for i in range(target.size(1)):
            source = torch.cat([source[:,i,:], attention], 1)
            output_decoder = self.decoder(source)
            attention = self.attention(output_decoder, output_encoder)
            output_decoders.append(output_decoder)

        output_decoders = torch.cat(output_decoders, 1)

        return output_decoders


