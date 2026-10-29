
import os
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from tqdm.notebook import tqdm

class InputEmbeddings(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size  = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x):
        return self.embeddings(x)

class RPositionalEmbeddings(nn.Module):
    def __init__(self, head_dim: int, seq_len: int, base: float= 10000.0):
        super().__init__()
        assert head_dim%2==0, "Head dim must be even."
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.base = base

        pow = -2*(torch.arange(0, head_dim//2).float())/head_dim
        theta = (base**pow)
        m = torch.arange(seq_len).float()
        #multiply each m with all theta
        angles = torch.outer(m, theta) #seq_len,head dim/2

        full_angle = torch.concat([angles, angles], dim=1) #seq len, head dim
        self.register_buffer("cos_table", torch.cos(full_angle), persistent=False)
        self.register_buffer("sin_table", torch.sin(full_angle), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        cos = self.cos_table[positions].unsqueeze(1) # batch size, 1, seq len, head dim
        sin = self.sin_table[positions].unsqueeze(1)
        # bs, heads, seq len, head dim
        out = (x*cos + torch.concat([-x[:,:,:,self.head_dim//2:], x[:,:,:,:self.head_dim//2]],\
                                     dim=-1)*sin).type_as(x)
        return out

class MultiHeadAttention(nn.Module):
    
    def __init__(self, dim: int, h_q: int, h_kv: int, dropout: float, rope: RPositionalEmbeddings):
        super().__init__()
        self.dim = dim
        self.h_q = h_q
        self.h_kv = h_kv
        self.repeats = h_q//h_kv
        self.positional_embed = rope
        
        self.d_k = dim//h_q
        self.w_q = nn.Linear(dim,self.d_k*h_q, bias= False)
        self.w_k = nn.Linear(dim,self.d_k*h_kv, bias= False)        
        self.w_v = nn.Linear(dim, self.d_k*h_kv, bias= False)  
        self.w_o = nn.Linear(dim, dim, bias= False)  
        self.dropout = nn.Dropout(dropout) 
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        scores = query@key.transpose(-2,-1)/math.sqrt(d_k)

        if mask is not None:
            scores.masked_fill_(mask==0, -1e9)
        scores = scores.softmax(dim=-1)
        
        ### do we need ?
        if dropout is not None:
            scores = dropout(scores)
            
        return scores@value,scores 
        
    def forward(self, q, k, v, pos,mask):
        # batch, seq len, embed dim -> batch, seq_len, d_k*h_q
        query = self.w_q(q)
        # batch, seq len, embed dim -> batch, seq_len, d_k*h_kv
        key = self.w_k(k)
        value = self.w_v(v)
        
        # batch, h_q, seq, d_k
        query = query.view(query.shape[0], query.shape[1], self.h_q, self.d_k).transpose(1,2)
        # batch, h_kv, seq, d_k
        key = key.view(key.shape[0], key.shape[1], self.h_kv, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h_kv, self.d_k).transpose(1,2)
        #apply positional embeddings to query and key
        query = self.positional_embed(query, pos)
        key = self.positional_embed(key, pos)
        if self.repeats!=1:
            # batch, h_q, seq, d_k
            key = key.repeat_interleave(repeats=self.repeats, dim=1)
            value = value.repeat_interleave(repeats=self.repeats, dim=1)

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,\
                                        dropout_p=self.dropout.p if self.training else 0.0, is_causal=mask is None)
        x = x.transpose(1,2).reshape(x.shape[0],-1, self.dim)
        return self.w_o(x)


class RMSNorm(nn.Module):

    def __init__(self, dim ,eps:float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        v = x.float()
        v = v*torch.rsqrt((v**2).mean(dim=-1, keepdim=True)+self.eps)
        v = v*self.weight.float()

        return v.type_as(x)


class FFN(nn.Module):
    def __init__(self, dim: int, ffn_multipler: Optional[int], multiple_of: int) -> None:
        super().__init__()
        hidden_dim = 4*dim
        hidden_dim = int(2*hidden_dim/3)
        if ffn_multipler is not None:
            hidden_dim = int(hidden_dim*ffn_multipler)
        #next multiple of multiple_of
        hidden_dim = multiple_of*((hidden_dim+multiple_of-1)//multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self,x):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish*x_V
        x = self.w2(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, args, rope: RPositionalEmbeddings):
        super().__init__()
        self.dim = args.dim
        self.n_heads_q = args.n_heads_q
        self.n_heads_kv = args.n_heads_kv if args.n_heads_kv is not None else args.n_heads_q
        assert self.n_heads_q % self.n_heads_kv == 0, "n_heads_q must be divisible by n_heads_kv"
        self.head_dim = args.dim // self.n_heads_q
        self.attention = MultiHeadAttention(self.dim, self.n_heads_q, self.n_heads_kv, args.dropout, rope)

        self.ffn = FFN(self.dim, ffn_multipler=args.ffn_multiplier, multiple_of=args.multiple_of)
        self.norm1 = RMSNorm(self.dim, args.norm_eps)
        self.norm2 = RMSNorm(self.dim, args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, pos, mask):
        # Attention block
        h = self.norm1(x)
        attn_out = self.attention(h, h, h, pos, mask)
        x = x + self.dropout(attn_out)

        # Feedforward block
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_blocks = nn.ModuleList()
        rope = RPositionalEmbeddings(args.dim//args.n_heads_q, args.max_seq_len)
        for _ in range(args.n_layers):
            self.encoder_blocks.append(EncoderBlock(args, rope))
        self.embeddings = InputEmbeddings(args.dim, args.vocab_size)
        self.projection = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.norm = RMSNorm(args.dim, args.norm_eps)

        self._init_weights(args.init_std)

    def forward(self, x, pos, mask):
        assert pos.max() < self.args.max_seq_len, "Position indices exceed maximum sequence length."
        x = self.embeddings(x)
        for block in self.encoder_blocks:
            x = block(x, pos, mask)
        x = self.norm(x)
        x = self.projection(x)
        return x

    def _init_weights(self, std):
        # every projection and the embedding table: normal(0, std).
        # RMSNorm gains are left at ones - matching on module type avoids them.
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=std)

        # each block writes to the residual stream twice (w_o and w2), so its
        # variance would grow like sqrt(2*n_layers) with depth. Scaling the two
        # residual-writing projections by the inverse keeps it depth-independent.
        scale = 1.0/math.sqrt(2*len(self.encoder_blocks))
        for block in self.encoder_blocks:
            block.attention.w_o.weight.data.mul_(scale)
            block.ffn.w2.weight.data.mul_(scale)

def _unwrap(m):
    m = getattr(m, 'module', m)      # DDP
    m = getattr(m, '_orig_mod', m)   # torch.compile
    return m


def save_model(model, optimizer, step, device, state_file_name, state_dir, loss):
    
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)

    if 'best' in state_file_name:
        state_path = os.path.join(state_dir, state_file_name + '.pth')
    else:   
        state_path = os.path.join(state_dir, state_file_name + '_' + str(step)+ '.pth')

    # make sure you transfer the model to cpu.
    if device == 'cpu':
        model.to('cpu')

    # save the state_dict
    torch.save({
        'model_state_dict': _unwrap(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': torch.tensor(step),
        'loss': torch.tensor(loss)
    }, state_path)
    
    #transfer model to gpu again
    if device == 'cuda':
        model.to('cuda')
    
    return

def load_model(model, state_file_name ,state_dir):
    state_path = os.path.join(state_dir, state_file_name)

    # loading the model and getting model parameters by using load_state_dict
    state = torch.load(state_path, map_location='cpu')
    _unwrap(model).load_state_dict(state['model_state_dict'])

    return model

