import torch


class Semiring:
    @classmethod
    def size(cls):
        return 1

    @classmethod
    def times(cls, *ls):
        cur = ls[0]
        for l in ls[1:]:
            cur = cls.mul(cur, l)
        return cur

    @classmethod
    def plus(cls, *ls):
        return cls.sum(torch.stack(ls), dim=0)

    @classmethod
    def dot(cls, *ls):
        return cls.sum(cls.times(*ls))

    @classmethod
    def convert(cls, potentials):
        return potentials.unsqueeze(0)

    @classmethod
    def unconvert(cls, potentials):
        return potentials.squeeze(0)


class _Base(Semiring):
    @staticmethod
    def mul(a, b):
        return torch.mul(a, b)

    @staticmethod
    def zero_(xs):
        return xs.fill_(0)

    @staticmethod
    def one_(xs):
        return xs.fill_(1)


class StdSemiring(_Base):
    @staticmethod
    def sum(xs, dim=-1):
        return torch.sum(xs, dim=dim)

    # @staticmethod
    # def div_exp(a, b):
    #     return a.exp().div(b.exp())

    @staticmethod
    def prod(a, dim=-1):
        return torch.prod(a, dim=dim)


class EntropySemiring(Semiring):
    # (z1, h1) ⊕ (z2, h2) = (logsumexp(z1,z2), h1 + h2), (65)
    # (z1, h1) ⊗ (z2, h2) = (z1 + z2, z1.exp()h2 + z2.exp()h1), (66)

    @staticmethod
    def size():
        return 2

    @staticmethod
    def convert(xs):
        values = torch.zeros((2,) + xs.shape).type_as(xs)
        values[0] = xs
        values[1] = 0
        return values

    @staticmethod
    def unconvert(xs):
        return xs[1]

    @staticmethod
    def sum(xs, dim=-1):
        assert dim != 0
        d = dim - 1 if dim > 0 else dim
        sm = torch.softmax(xs[0], dim=d)
        return torch.stack(
            (
                torch.logsumexp(xs[0], dim=d),
                torch.sum(xs[1].mul(sm) - sm.log().mul(sm), dim=d),
            )
        )

    @staticmethod
    def mul(a, b):
        return torch.stack((a[0] + b[0], a[1] + b[1]))

    @classmethod
    def prod(cls, xs, dim=-1):
        return xs.sum(dim)
        # assert dim!=0 and xs.dim()-dim != 0

        # if dim < 0: dim = xs.dim() + dim
        # values = torch.zeros(*((s if x != dim else 1) for (x, s) in enumerate(xs.shape)) ).type_as(xs)
        # cls.one_(values)
        # for i in torch.arange(xs.shape[dim]):
        #     values = cls.mul(values, xs.index_select(dim, i))
        # values =  values.squeeze(dim)
        # return values

    @staticmethod
    def zero_(xs):
        xs[0].fill_(-1e9)
        xs[1].fill_(0)
        return xs

    @staticmethod
    def one_(xs):
        xs[0].fill_(0)
        xs[1].fill_(0)
        return xs


class _BaseLog(Semiring):
    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def zero_(xs):
        return xs.fill_(-1e9)

    @staticmethod
    def one_(xs):
        return xs.fill_(0.0)

    # @staticmethod
    # def div_exp(a, b):
    #     return (a - b).exp()

    @staticmethod
    def prod(a, dim=-1):
        return torch.sum(a, dim=dim)


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
