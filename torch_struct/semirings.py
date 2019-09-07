import torch


class Semiring:
    @classmethod
    def times(cls, *ls):
        cur = ls[0]
        for l in ls[1:]:
            cur = cls.mul(cur, l)
        return cur

    @classmethod
    def dot(cls, *ls):
        return cls.sum(cls.times(*ls))


class _Base(Semiring):
    @staticmethod
    def mul(a, b):
        return torch.mul(a, b)

    @staticmethod
    def zero():
        return 0

    @staticmethod
    def one():
        return 1


class StdSemiring(_Base):
    @staticmethod
    def sum(xs, dim=-1):
        return torch.sum(xs, dim=dim)

    def div_exp(a, b):
        return a.exp().div(b.exp())

class _BaseLog(Semiring):
    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def zero():
        return -1e9

    @staticmethod
    def one():
        return 0.0

    @staticmethod
    def div_exp(a, b):
        return a.exp().div(b.exp())


class LogSemiring(_BaseLog):
    @staticmethod
    def sum(xs, dim=-1):
        return torch.logsumexp(xs, dim=dim)


class MaxSemiring(_BaseLog):
    @staticmethod
    def sum(xs, dim=-1):
        return torch.max(xs, dim=dim)[0]

    @staticmethod
    def div_exp(a, b):
        return a == b


class _SampledLogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.save_for_backward(input)
        ctx.dim = dim
        xs = input
        d = torch.max(xs, keepdim=True, dim=dim)[0]
        return torch.log(torch.sum(torch.exp(xs - d), dim=dim)) + d.squeeze(dim)

    @staticmethod
    def backward(ctx, grad_output):
        logits, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            s = torch.distributions.OneHotCategorical(logits=logits).sample()
            grad_input = grad_output.unsqueeze(ctx.dim).mul(s)
        return grad_input, None


class SampledSemiring(_BaseLog):
    @staticmethod
    def sum(xs, dim=-1):
        return _SampledLogSumExp.apply(xs, dim)
