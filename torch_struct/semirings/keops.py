import torch
import torch.distributions
from .semirings import _BaseLog

try:
    from pykeops.torch import LazyTensor
except ImportError:
    pass


class LogSemiringKO(_BaseLog):
    """
    Implements the log-space semiring (logsumexp, +, -inf, 0).

    Gradients give marginals.
    """

    @staticmethod
    def sum(a, dim=-1):
        a_lazy = LazyTensor(a.unsqueeze(-1).unsqueeze(-1).contiguous())
        c = a_lazy.sum(-1).logsumexp(a.dim() - 1).squeeze(-1).squeeze(-1)
        return c

    @classmethod
    def dot(cls, a, b):
        """
        Dot product along last dim. (Faster than calling sum and times.)
        """
        a_lazy = LazyTensor(a.unsqueeze(-1).unsqueeze(-1).contiguous())
        b_lazy = LazyTensor(b.unsqueeze(-1).unsqueeze(-1).contiguous())
        c = (a_lazy + b_lazy).sum(-1).logsumexp(a.dim() - 1).squeeze(-1).squeeze(-1)
        return c


class _Max(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        one_hot = b.shape[-1]
        a_lazy = LazyTensor(a.unsqueeze(-1).unsqueeze(-1).contiguous())
        b_lazy = LazyTensor(b.unsqueeze(-1).unsqueeze(-1).contiguous())
        c = (a_lazy + b_lazy).sum(-1).max(a.dim() - 1).squeeze(-1).squeeze(-1)
        ac = (a_lazy + b_lazy).sum(-1).argmax(a.dim() - 1).squeeze(-1).squeeze(-1)
        ctx.save_for_backward(ac, torch.tensor(one_hot))
        return c

    @staticmethod
    def backward(ctx, grad_output):
        ac, size = ctx.saved_tensors
        back = torch.nn.functional.one_hot(ac, size).type_as(grad_output)
        ret = grad_output.unsqueeze(-1).mul(back)
        return ret, ret


class MaxSemiringKO(_BaseLog):
    @classmethod
    def sum(cls, xs, dim=-1):
        assert dim == -1
        return cls.dot(xs, xs.clone().fill_(0))

    @classmethod
    def dot(cls, a, b):
        """
        Dot product along last dim. (Faster than calling sum and times.)
        """
        return _Max.apply(a, b)
