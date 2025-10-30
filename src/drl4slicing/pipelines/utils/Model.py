import pfrl
import math
import torch
import numpy as np
import pfrl.initializers
import torch.nn.functional as F

#=========================================================================================

def lecun_init(layer, gain=1):
    if isinstance(layer, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        torch.nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        torch.nn.init.zeros_(layer.bias_ih_l0)
        torch.nn.init.zeros_(layer.bias_hh_l0)
    return layer

#=========================================================================================
#=========================================================================================
#=========================================================================================
#=========================================================================================

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(embed_dim, embed_dim)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # Project inputs to query, key, and value
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = self.softmax(scores)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attn_output = self.out(attn_output)

        # apply dropout
        attn_output = self.dropout_module(attn_output)
        
        return attn_output


class Encoder(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, feed_forward_multiplicator=8): # ff_size_mul = feed_forward_multiplicator
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.multihead_attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        # self.multihead_attn = CausalSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm1 = torch.nn.LayerNorm(embed_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, feed_forward_multiplicator * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(feed_forward_multiplicator * embed_dim, embed_dim),
            torch.nn.Dropout(dropout)
        )
        self.layer_norm2 = torch.nn.LayerNorm(embed_dim)
        

    def forward(self, x, mask=None):
        attn_output = self.multihead_attn(x, mask=mask)[0]
        x = self.layer_norm1(x + attn_output)
        # x = x + attn_output
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        # x = x + ff_output
        return x


class AttentionQFunction_D_DQN(torch.nn.Module):
    def __init__(self, n_encoders, embed_dim, num_heads, dropout, ff_size_mul, hidden_sizes, n_actions) -> None:
        super(AttentionQFunction_D_DQN, self).__init__()
        self.encoders = torch.nn.ModuleList()
        for _ in range(n_encoders):
            self.encoders.append( Encoder(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, feed_forward_multiplicator=ff_size_mul) )

        self.learner_head = torch.nn.Sequential(
            pfrl.nn.MLP(in_size=embed_dim * n_actions,
                        hidden_sizes=hidden_sizes,
                        nonlinearity=torch.relu,
                        out_size=n_actions,
                        last_wscale=1),
            pfrl.q_functions.DiscreteActionValueHead(),
        )
    

    def forward(self, x):
        # extract features using stacked encoders
        for encoder in self.encoders:
            x = encoder(x)
        x = x.view(x.size(0), -1)
        # main fully connected neural network (fcnn)
        x = self.learner_head(x)
        return x

#=========================================================================================
#=========================================================================================
#=========================================================================================
#=========================================================================================

if __name__=='__main__':
    pass
