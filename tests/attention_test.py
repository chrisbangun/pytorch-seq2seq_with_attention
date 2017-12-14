import unittest

import numpy as np
import torch
import torch.nn as nn

from modules.Attention import BahdanauAttention
from torch.autograd import Variable


class AttentionTest(unittest.TestCase):

    def test_bahdanau_attention(self):
        hidden_size = 128
        query_size = (3, 128)
        memory_size = (3, 15, 128)
        
        context_size = (memory_size[0], memory_size[2])
        alignment_size = (memory_size[0], memory_size[1])

        query = Variable(
            torch.from_numpy(np.random.randn(*query_size).astype(np.float32)))
        keys = Variable(
            torch.from_numpy(np.random.randn(*memory_size).astype(np.float32)))
        
        bahdanau_attention = BahdanauAttention(
            hidden_size=hidden_size,
            query_size=query_size[1],
            memory_size=memory_size[-1])

        context, alignment_score = bahdanau_attention(query, keys)
        self.assertEqual(context.size(), context_size, "context must has the same size")
