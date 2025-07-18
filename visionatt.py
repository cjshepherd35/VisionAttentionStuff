import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

#for now
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout=0.2
print(device)
class Head(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.key = nn.Linear(input_size, hidden_size, bias=False)
        self.query = nn.Linear(input_size, hidden_size, bias=False)
        self.value = nn.Linear(input_size, hidden_size, bias=False)
        self.ffwd = nn.Linear(hidden_size, input_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b,t,c = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attention scores
        wei = q @ k.transpose(-2,-1) * c**-0.5
        # wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf')) #(b,t,t)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        #perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        out = self.ffwd(out)
        return out
    


class FeedForward(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 4*n_inputs),
            nn.ReLU(), 
            nn.Linear(4*n_inputs, n_inputs), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        # head_size = n_embed // n_head
        self.sa = Head(n_inputs, n_hidden)
        self.ffwd = FeedForward(n_inputs)
        self.ln1 = nn.LayerNorm(n_inputs)
        self.ln2 = nn.LayerNorm(n_inputs)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    


class Transform(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, block_size, n_layers):
        super().__init__()
        self.position_embedding_table = nn.Embedding(block_size, n_inputs)
        self.blocks = nn.Sequential(*[Block(n_inputs, n_hidden) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_inputs)
        self.lm_head = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        b,t,ins = x.shape
        pos_embed = self.position_embedding_table(torch.arange(t, device=device))
        xout = x + pos_embed

        xout = self.blocks(xout)
        xout = self.ln_f(xout)
        # print('before ', xout.shape)
        output = self.lm_head(xout.mean(dim=1))
        # print('out ', output)
        # output = F.sigmoid(output)
        # print('probs ', output)   
          
        return output
    

