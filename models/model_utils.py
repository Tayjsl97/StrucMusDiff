import math, copy

import ipdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + Variable(self.pe[:, :x.size(1)],
        #                  requires_grad=False)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # print("q,k,v,mask: ",query.shape,key.shape,value.shape,mask.shape)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # print("attn: ",scores,math.sqrt(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # print("score: ",scores.shape,", mask: ",mask.shape)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # print(query.shape, key.shape, value.shape)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.dropout=dropout
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.linear1 = nn.Linear(size, size*4)
        self.linear2 = nn.Linear(size*4, size)
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # x=self.norm1(x+self.dropout(self.self_attn(x, x, x, mask)))
        # y = self.dropout(F.gelu(self.linear1(x)))
        # y = self.dropout(self.linear2(y))
        # return self.norm2(x+y)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, latent_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attn = self_attn
        self.latent_attn = latent_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, z, tgt_mask, tgt_tri_mask):
        "Follow Figure 1 (right) for connections."
        self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_tri_mask))
        self.sublayer[1](x, lambda x: self.latent_attn(x, z, z, tgt_mask))
        return self.sublayer[2](x, self.feed_forward)


def full_mask(batch, seqLen1, seqLen2):
    "Mask out subsequent positions."
    attn_shape = (batch, seqLen1, seqLen2)
    encoder_mask = np.ones(attn_shape)
    return (torch.from_numpy(encoder_mask)).to(device)


def triple_mask(batch,l_decoder_seq):
    decoder_attn_shape = (batch, l_decoder_seq, l_decoder_seq)
    decoder_mask = np.triu(np.ones(decoder_attn_shape), k=1).astype('uint8')
    return (torch.from_numpy(decoder_mask) == 0).to(device)


def length_mask(batch, full_length, length):
    "Mask out subsequent positions."
    mask=torch.zeros((batch,full_length,full_length)).to(device)
    for i in range(batch):
        mask[i][:length[i]]=torch.cat((torch.ones(length[i],length[i]),torch.zeros(length[i],full_length-length[i])),dim=-1)
        mask[i][length[i]:] = torch.cat((torch.zeros(full_length-length[i],length[i]),
                                        torch.ones(full_length-length[i], full_length-length[i])), dim=-1)
    return mask


def length_tri_mask(batch, full_length, length):
    "Mask out subsequent positions."
    mask=torch.zeros((batch,full_length,full_length)).to(device)
    for i in range(batch):
        attn_shape = (length[i], length[i])
        tri_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        tri_mask=(torch.from_numpy(tri_mask) == 0).to(device)
        right_upper=torch.zeros(length[i],full_length-length[i]).to(device)
        left_lower=right_upper.t()
        right_lower=torch.ones(full_length-length[i],full_length-length[i]).to(device)
        mask[i][:length[i]]=torch.cat((tri_mask,right_upper),dim=-1)
        mask[i][length[i]:] = torch.cat((left_lower,right_lower),dim=-1)
    return mask


def loss_mask_func(batch,length):
    mask = torch.zeros((batch, 1023)).to(device)
    for i in range(batch):
        mask[i][:length[i]-2]=torch.ones(length[i]-2).to(device)
        mask[i][length[i] - 2:] = torch.zeros(1023-length[i]+2).to(device)
    return mask


def gumbel_softmax(logits, temperature):
    return F.gumbel_softmax(logits, tau=temperature, hard=True)


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


# -- nucleus -- #
def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    try:
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    except:
        ipdb.set_trace()
    return word


def sampling(logit, p=None, t=1.0, is_training=False):
    if is_training:
        logit = logit.squeeze()
        probs = gumbel_softmax(logits=logit, temperature=t)

        return torch.argmax(probs)

    else:
        logit = logit.squeeze().cpu().numpy()
        probs = softmax_with_temperature(logits=logit, temperature=t)

        if probs is None:
            return None

        if p is not None:
            cur_word = nucleus(probs, p=p)

        else:
            cur_word = weighted_sampling(probs)
        return cur_word


def softmax_with_temperature(logits, temperature):
    c = np.max(logits)
    probs = np.exp((logits-c) / temperature) / np.sum(np.exp((logits-c) / temperature))
    if np.isnan(probs).any():
        return None
    else:
        return probs
