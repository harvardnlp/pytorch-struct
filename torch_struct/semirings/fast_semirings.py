import torch
import torch.distributions
from .semirings import _BaseLog
import genmatmul

def matmul_size(a, b):
    size = [max(i,j) for i, j in zip(a.shape[:-2], b.shape[:-2])]
    size.append(a.shape[-2])
    size.append(b.shape[-1])
    return size


class FastLogSemiring(_BaseLog):
    """
    Implements the log-space semiring (logsumexp, +, -inf, 0).

    Gradients give marginals.
    """

    @staticmethod
    def matmul(a, b, dims=1):
        size = matmul_size(a, b)
        a = a.expand(*size[:-2], a.shape[-2], a.shape[-1])
        b = b.expand(*size[:-2], b.shape[-2], b.shape[-1])
        a2 = a.contiguous().view(-1, a.shape[-2], a.shape[-1])
        b2 = b.contiguous().view(-1, b.shape[-2], b.shape[-1])
        return genmatmul.logmatmul(a2, b2).view(size)
