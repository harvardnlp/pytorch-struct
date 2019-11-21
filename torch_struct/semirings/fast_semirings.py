import torch
import torch.distributions
from .semirings import _BaseLog
from .sample import _SampledLogSumExp

try:
    import genbmm
except ImportError:
    pass


def matmul_size(a, b):
    size = [max(i, j) for i, j in zip(a.shape[:-2], b.shape[:-2])]
    size.append(a.shape[-2])
    size.append(b.shape[-1])
    return size


def broadcast(a, b):
    size = matmul_size(a, b)
    a = a.expand(*size[:-2], a.shape[-2], a.shape[-1])
    b = b.expand(*size[:-2], b.shape[-2], b.shape[-1])
    a2 = a.contiguous().view(-1, a.shape[-2], a.shape[-1])
    b2 = b.contiguous().view(-1, b.shape[-2], b.shape[-1])
    return a2, b2, size


class FastLogSemiring(_BaseLog):
    """
    Implements the log-space semiring (logsumexp, +, -inf, 0).

    Gradients give marginals.
    """

    @staticmethod
    def sum(xs, dim=-1):
        return torch.logsumexp(xs, dim=dim)

    @staticmethod
    def matmul(a, b, dims=1):
        if isinstance(a, genbmm.BandedMatrix):
            return b.multiply_log(a.transpose())
        else:
            a2, b2, size = broadcast(a, b)
            return genbmm.logbmm(a2, b2).view(size)


class FastMaxSemiring(_BaseLog):
    @staticmethod
    def sum(xs, dim=-1):
        return torch.max(xs, dim=dim)[0]

    @staticmethod
    def matmul(a, b, dims=1):
        a2, b2, size = broadcast(a, b)
        return genbmm.maxbmm(a2, b2).view(size)


class FastSampleSemiring(_BaseLog):
    @staticmethod
    def sum(xs, dim=-1):
        return _SampledLogSumExp.apply(xs, dim)

    @staticmethod
    def matmul(a, b, dims=1):
        a2, b2, size = broadcast(a, b)
        return genbmm.samplebmm(a2, b2).view(size)
