import torch
import torch.nn as nn
import math, copy, time
from models.model_utils import PositionalEncoding
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask, FullMask


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Reference_Phrase_Encoder(nn.Module):
    def __init__(self,N,H,width,dropout,activate,attention_type):
        super(Reference_Phrase_Encoder,self).__init__()
        c = copy.deepcopy
        self.n_layer = N
        self.n_head = H
        self.width=width
        self.input_linear=nn.Linear(width,width)
        self.pos_embedd = nn.Sequential(c(PositionalEncoding(self.width, dropout)))
        self.transformer = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.n_layer,
            n_heads=self.n_head,
            query_dimensions=self.width // self.n_head,
            value_dimensions=self.width // self.n_head,
            feed_forward_dimensions=self.width * 4,
            activation=activate,
            dropout=dropout,
            attention_type=attention_type,
        ).get()

    def forward(self,input_x,input_y=None):
        input_data = self.input_linear(input_x)
        input_data = self.pos_embedd(input_data)
        mask = FullMask(None, input_data.shape[1], input_data.shape[1], device=input_data.device)
        # mask = TriangularCausalMask(input_data.shape[1], device=input_data.device)
        output = self.transformer(input_data, mask)
        return output







