""" Patch Embedding Module. """
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce  # Use to insert into the model definition as a layer.
from torchsummary import summary


class PatchEmbedding(nn.Module):
    """ Version that uses nn.Conv2d instead of a nn.Linear. Apparently this is more computationally efficient? Idk. """
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768, # channel x patch_size x patch_size
                 img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size

        # Initialize pieces of the model. (Hybrid Architecture, as described in the paper (Sec 3.1))
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),        # Increase the number of channels from 3 to emb_size.
            Rearrange('b e (h) (w) -> b (h w) e'))                                              # Linearize. (h w): combined dim  e: emb_size
        self.clf_token = nn.Parameter(torch.randn(1, 1, emb_size))                              # Trainable parameter; denoted as * in the paper.
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, emb_size))   # Trainable parameter.

    def forward(self, x: Tensor, verbose=False) -> Tensor:
        b, _, _, _ = x.shape  # Get batch size.
        x = self.projection(x)
        if verbose:
            print("x:\t\t", x.shape)

        clf_tokens = repeat(self.clf_token, '() n e -> b n e', b=b)  # [BS, 1, 768]; prepend all patches. Useful einops broadcasting trick.
        if verbose:
            print("clf_tokens:\t", clf_tokens.shape)

        x = torch.cat([clf_tokens, x], dim=1)  # Prepend the clf token to the input
        if verbose:
            print("x:\t\t", x.shape)

        x += self.positions  # Add position embedding
        if verbose:
            print("positions:\t", self.positions.shape)

        return x

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
        queries, keys, values = qkv[0], qkv[1], qkv[2] # Separate after making the matmul pass -> more computationally efficient.

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

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 expansion: int = 4, # Where is this from?
                 drop_p: float = 0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
    



if __name__=='__main__':
    print('Hi')
