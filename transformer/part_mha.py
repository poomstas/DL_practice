""" Multi-Head Attention (MHA) Module """
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, reduce, repeat

from model import PatchEmbedding
from torchsummary import summary

# emb_size = 768
# num_heads = 8

# keys = nn.Linear(emb_size, emb_size)
# queries = nn.Linear(emb_size, emb_size)
# values = nn.Linear(emb_size, emb_size)

# PE = PatchEmbedding()

# x = torch.randn((99, 3, 224, 224)) # 99 is the batch size
# x = PE(x, verbose=False)

# print(queries(x).shape)  # Same dimension as the original x
# queries = rearrange(queries(x), 'b n (h d) -> b h n d', h=num_heads)  # batch, head, n, emb_size/head
# keys = rearrange(keys(x), 'b n (h d) -> b h n d', h=num_heads)
# values = rearrange(values(x), 'b n (h d) -> b h n d', h=num_heads)
# print('shapes:\t', queries.shape)  # Same dimension as the original x
# print('shapes:\t', keys.shape)  # Same dimension as the original x
# print('shapes:\t', values.shape)  # Same dimension as the original x

# # Question: Why separate (h d)? by number of heads? -> also does the rearrange at the end as well.


# # Queries * Keys
# energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # qd, kd -> qk (involves matrix transpose (before doing matmul))
# print('energy :', energy.shape)

# # Get Attention Score
# scaling = emb_size ** (1/2)
# att = F.softmax(energy/scaling, dim=-1) 
# print('att :', att.shape)

# # Attention Score * values
# out = torch.einsum('bhal, bhlv -> bhav ', att, values) # al, lv -> av, while having batch_size and num_heads constant.
# print('out :', out.shape)

# # Rearrage to emb_size
# out = rearrange(out, "b h n d -> b n (h d)")
# print('out2 : ', out.shape)

class MultiHeadAttention(nn.Module):
    def __init__(self,
                 emb_size: int = 768,
                 num_heads: int = 8,
                 dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        # Fuse the queries, keys and values in one matrix. -> Nice detail.
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x: Tensor, mask: Tensor=None):
        # Split the keys, queries, and values in num_heads.
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d",
                        h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2] # Separate after making the matmul pass.

        # Sum over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len.

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy/scaling, dim=1)
        att = self.att_drop(att)

        # Sum over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', att, values)  # batch, num_heads, attention, l(probably just a placeholder?), values
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out

if __name__=='__main__':
    PE = PatchEmbedding()
    MHA = MultiHeadAttention()

    x = torch.randn(8, 3, 224, 224)
    print(x.shape)
    x = PE(x)
    print(x.shape)
    x = MHA(x)
    print(x.shape)

    x = torch.randn(8, 197, 768)  # Dims you get after putting through PatchEmbedding.
    summary(MHA, x.shape[1:], device='cpu')
    