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

if __name__=='__main__':
    PE = PatchEmbedding()
    summary(PE, (3, 224, 224), device='cpu')

    input = torch.randn((8, 3, 224, 224))
    out = PE(input, verbose=True)
    print("out:\t\t", out.shape)
