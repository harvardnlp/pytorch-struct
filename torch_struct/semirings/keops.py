import torch
import torch.distributions
from pykeops.torch import LazyTensor
from .semirings import _BaseLog

class LogSemiringKO(_BaseLog):
    """
    Implements the log-space semiring (logsumexp, +, -inf, 0).

    Gradients give marginals.
    """

    @staticmethod
    def sum(xs, dim=-1):
        return torch.logsumexp(xs, dim=dim)

    @classmethod
    def dot(cls, a, b):
        """
        Dot product along last dim. (Faster than calling sum and times.)
        """
        a_lazy = LazyTensor(a.unsqueeze(-1).unsqueeze(-1).contiguous())
        b_lazy = LazyTensor(b.unsqueeze(-1).unsqueeze(-1).contiguous())
        c = (a_lazy + b_lazy).sum(-1).logsumexp(a.dim()-1).squeeze(-1).squeeze(-1)
        return c

class MaxSemiringKO(_BaseLog):
    @staticmethod
    def sum(xs, dim=-1):
        return torch.max(xs, dim=dim)[0]

    @classmethod
    def dot(cls, a, b):
        """
        Dot product along last dim. (Faster than calling sum and times.)
        """
        a_lazy = LazyTensor(a.unsqueeze(-1).unsqueeze(-1).contiguous())
        b_lazy = LazyTensor(b.unsqueeze(-1).unsqueeze(-1).contiguous())
        c = (a_lazy + b_lazy).sum(-1).max(a.dim()-1).squeeze(-1).squeeze(-1)
        return c
