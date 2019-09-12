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
        ctx.save_for_backward(input, torch.tensor(dim))
        return torch.logsumexp(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        logits, dim = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            def sample(ls):
                pre_shape = ls.shape
                draws = torch.multinomial(ls.softmax(-1).view(-1, pre_shape[-1]), 1, True)
                draws.squeeze(1)
                return torch.nn.functional.one_hot(draws, pre_shape[-1]).view(*pre_shape).type_as(ls)

            if dim == -1:
                s=sample(logits)
            else:
                dim = dim if dim >= 0 else logits.dim() + dim
                perm = [i for i in range(logits.dim()) if i != dim] + [dim]
                rev_perm = [a for a,b in sorted(enumerate(perm), key=lambda a:a[1])]
                s= sample(logits.permute(perm)).permute(rev_perm)

            grad_input = grad_output.unsqueeze(dim).mul(s)
        return grad_input, None


class SampledSemiring(_BaseLog):
    @staticmethod
    def sum(xs, dim=-1):
        return _SampledLogSumExp.apply(xs, dim)


bits = torch.tensor([pow(2, i) for i in range(1, 18)])

class _MultiSampledLogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        part = torch.logsumexp(input, dim=dim)
        ctx.save_for_backward(input, part, torch.tensor(dim))
        return part

    @staticmethod
    def backward(ctx, grad_output):
        assert ((grad_output == 64) + (grad_output == 0) + (grad_output ==1)).all()

        logits, part, dim = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            def sample(ls):
                pre_shape = ls.shape
                draws = torch.multinomial(ls.softmax(-1).view(-1, pre_shape[-1]), 16, True)
                draws.transpose(0, 1)
                return torch.nn.functional.one_hot(draws, pre_shape[-1]).view(16, *pre_shape).type_as(ls)

            if dim == -1:
                s = sample(logits)
            else:
                dim = dim if dim >= 0 else logits.dim() + dim
                perm = [i for i in range(logits.dim()) if i != dim] + [dim]
                rev_perm =[0] + [a+1 for a,b in sorted(enumerate(perm), key=lambda a:a[1])]
                s= sample(logits.permute(perm)).permute(rev_perm)


            dim = dim if dim >= 0 else logits.dim() + dim
            final = (grad_output % 2).unsqueeze(0)
            mbits = bits[:].type_as(grad_output)
            on = grad_output.unsqueeze(0) % mbits.view(17, * [1]*grad_output.dim())
            on = on[1:] - on[:-1]
            old_bits = (on + final == 0).unsqueeze(dim+1)
            grad_input_check = mbits[0] * s[0].masked_fill_(old_bits[0], 0) + \
                               mbits[5] * s[5].masked_fill_(old_bits[5], 0)

            grad_input = mbits[:-1].view(16, *[1]*(s.dim()-1)).mul(
                                   s.masked_fill_(old_bits,0))
            #assert (grad_input_check == grad_input[5]).all()
            #final = (grad_output % 2).unsqueeze(0)
            #on = grad_output.unsqueeze(0) % mbits.view(17, * [1]*grad_output.dim())
            #on = on[1:] - on[:-1]
            #old_bits = (on + final != 0).unsqueeze(dim+1)
            #assert grad_input.shape == logits.shape
        return grad_input_check, None #torch.sum(grad_input, dim=0), None


class MultiSampledSemiring(_BaseLog):
    @staticmethod
    def sum(xs, dim=-1):
        return _MultiSampledLogSumExp.apply(xs, dim)

    @staticmethod
    def to_discrete(xs, j):
        i = j
        final = xs % 2
        mbits = bits.type_as(xs)
        return ((xs % mbits[i + 1] - xs % mbits[i]  + final)!= 0).type_as(xs)
