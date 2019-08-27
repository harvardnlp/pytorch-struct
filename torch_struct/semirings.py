import torch
import sys
import opt_einsum as oe


class Semiring:
    @classmethod
    def tensordot(cls, a, b, axes):
        "Generalized tensor dot product for semiring."
        def shape(shape, dot):
            a_shape, a_end = [], []
            for i, d in enumerate(shape):
                if i in dot:
                    a_end.append(i)
                else:
                    a_shape.append(i)
            return a_shape, a_end
            a = a.permute(*(a_shape + a_end))
        a_shape, a_end = shape(a.shape, axes[0])
        b_shape, b_end = shape(b.shape, axes[1])

        a = a.permute(*(a_shape + a_end))
        b = b.permute(*(b_shape + b_end))
        a = a.view(a.shape[:len(a_shape)] + (1,) * len(b_shape) + a.shape[len(a_shape):])
        b = b.view((1,) * len(a_shape) + b.shape[:len(b_shape)] + b.shape[len(b_shape):])
        c = (cls.mul(a, b)).contiguous()
        c = c.view(a.shape[:len(a_shape)] + b.shape[:len(b_shape)] + (-1,))
        return cls.sum(c).squeeze(-1)

    @classmethod
    def contract(cls, term, *vals):
        return oe.contract(term, *vals, backend=cls.name)

    @staticmethod
    def transpose(x, axes):
        return x.permute(*axes)


    @staticmethod
    def _register(semiring):
        sys.modules[semiring.name] = semiring


class _Base(Semiring):
    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def zero():
        return 0

    @staticmethod
    def one():
        return 1

class StdSemiring(_Base):
    name = "Semiring.Std"
    @staticmethod
    def sum(self, xs, dim=-1):
        return torch.sum(xs, dim=dim)
Semiring._register(StdSemiring)

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


class LogSemiring(_BaseLog):
    name = "Semiring.Log"
    @staticmethod
    def sum(xs, dim=-1):
        return torch.logsumexp(xs, dim=dim)
Semiring._register(LogSemiring)


class MaxSemiring(_BaseLog):
    name = "Semiring.Max"
    def plus(self, xs, dim=-1):
        return torch.max(xs, dim=dim)[0]
Semiring._register(MaxSemiring)


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
        grad_input =  None
        if ctx.needs_input_grad[0]:
            s = torch.distributions.OneHotCategorical(logits=logits).sample()
            grad_input = grad_output.unsqueeze(ctx.dim).mul(s)
        return grad_input, None

class SampledSemiring(_BaseLog):
    name = "Semiring.Sampled"
    def plus(self, xs, dim=-1):
        return _SampledLogSumExp.apply(xs, dim)
Semiring._register(SampledSemiring)
