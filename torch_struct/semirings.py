import torch
import sys
#import opt_einsum as oe


class Semiring:
    # @classmethod
    # def tensordot(cls, a, b, axes):
    #     "Generalized tensor dot product for semiring."
    #     print(a.shape, b.shape, axes)
    #     for i, j in zip(*axes):
    #         if i == -1:
    #             b = b.sum(dim=j, keepdim=True)
    #         if j == -1:
    #             a = a.sum(dim=i, keepdim=True)

    #     def shape(shape, dot):
    #         a_shape, a_end = [], []
    #         d = dict([(d, i) for i, d in enumerate(dot)])
    #         coda = [1,] * len(dot)
    #         for i, di in enumerate(shape):
    #             if i in d:
    #                 coda[d[i]] = di
    #                 a_end.append(i)
    #             else:
    #                 a_shape.append(i)
    #         return a_shape, a_end, coda
    #     a_shape, a_end, a_coda = shape(a.shape, axes[0])
    #     b_shape, b_end, b_coda = shape(b.shape, axes[1])
    #     a = a.permute(*(a_shape + a_end))
    #     b = b.permute(*(b_shape + b_end))
    #     a = a.view(
    #         a.shape[: len(a_shape)] + (1,) * len(b_shape) + tuple(a_coda)
    #     )
    #     b = b.view(
    #         (1,) * len(a_shape) + b.shape[: len(b_shape)] + tuple(b_coda)
    #     )
    #     c = (cls.mul(a, b)).contiguous()
    #     c = c.view(c.shape[: len(a_shape)] + c.shape[len(a_shape): len(a_shape + b_shape)] + (-1,))
    #     c = cls.sum(c)
    #     print(c.shape)
    #     assert(c.dim() == len(a_shape) + len(b_shape))
    #     return c

    # @classmethod
    # def contract(cls, term, *vals):
    #     return oe.contract(term, *vals, backend=cls.name)

    # @staticmethod
    # def transpose(x, axes):
    #     return x.permute(*axes)

    @staticmethod
    def _register(semiring):
        pass
        # sys.modules[semiring.name] = semiring

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
    @staticmethod
    def sum(xs, dim=-1):
        return torch.logsumexp(xs, dim=dim)




class MaxSemiring(_BaseLog):
    @staticmethod
    def sum(xs, dim=-1):
        return torch.max(xs, dim=dim)[0]




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
