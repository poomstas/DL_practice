import torch
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

if __name__=='__main__':
    # Transpose
    a = torch.arange(6).reshape(3, 2)
    print(a)
    print(torch.einsum('ij -> ji', [a]))

    # Sum all elements
    print('='*80)
    print(a)
    print(torch.einsum('ij->', [a]))

    # Column sum
    print('='*80)
    print(a)
    print(torch.einsum('ij->i', [a]))

    # Row sum
    print('='*80)
    print(a)
    print(torch.einsum('ij->j', [a]))

    # Matrix-Vector Multiplication
    print('='*80)
    a = torch.arange(6).reshape(2, 3)
    b = torch.arange(3)
    print(a)
    print(b)
    print(torch.einsum('ij,j->i', [a, b]))

    # Matrix-Matrix Multiplication
    print('='*80)
    a = torch.arange(9).reshape(3, 3)
    b = torch.arange(6).reshape(3, 2)
    print(a)
    print(b)
    print(torch.einsum('ik,kj->ij', [a, b]))

    # Dot Product - Vector
    print('='*80)
    a = torch.arange(3)
    b = torch.arange(3, 6)
    print(a)
    print(b)
    print(torch.einsum('i,j->', [a, b]))

    # Dot Product - Matrix
    print('='*80)
    a = torch.arange(6).reshape(2, 3)
    b = torch.arange(6, 12).reshape(2, 3)
    print(a)
    print(b)
    print(torch.einsum('ij,ij->', [a, b]))
