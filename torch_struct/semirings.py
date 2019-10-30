import torch
import torch.distributions


class Semiring:
    """
    Base semiring class.

    Based on description in:

    * Semiring parsing :cite:`goodman1999semiring`

    """

    @classmethod
    def size(cls):
        "Additional *ssize* first dimension needed."
        return 1

    @classmethod
    def dot(cls, *ls):
        "Dot product along last dim."
        return cls.sum(cls.times(*ls))

    @classmethod
    def times(cls, *ls):
        "Multiply a list of tensors together"
        cur = ls[0]
        for l in ls[1:]:
            cur = cls.mul(cur, l)
        return cur

    @classmethod
    def convert(cls, potentials):
        "Convert to semiring by adding an extra first dimension."
        return potentials.unsqueeze(0)

    @classmethod
    def unconvert(cls, potentials):
        "Unconvert from semiring by removing extra first dimension."
        return potentials.squeeze(0)

    @staticmethod
    def zero_(xs):
        "Fill *ssize x ...* tensor with additive identity."
        raise NotImplementedError()

    @staticmethod
    def one_(xs):
        "Fill *ssize x ...* tensor with multiplicative identity."
        raise NotImplementedError()

    @staticmethod
    def sum(xs, dim=-1):
        "Sum over *dim* of tensor."
        raise NotImplementedError()

    @classmethod
    def plus(cls, a, b):
        return cls.sum(torch.stack([a, b], dim=-1))


class _Base(Semiring):
    @staticmethod
    def mul(a, b):
        return torch.mul(a, b)

    @staticmethod
    def prod(a, dim=-1):
        return torch.prod(a, dim=dim)

    @staticmethod
    def zero_(xs):
        return xs.fill_(0)

    @staticmethod
    def one_(xs):
        return xs.fill_(1)


class _BaseLog(Semiring):
    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def zero_(xs):
        return xs.fill_(-1e5)

    @staticmethod
    def one_(xs):
        return xs.fill_(0.0)

    @staticmethod
    def prod(a, dim=-1):
        return torch.sum(a, dim=dim)


class StdSemiring(_Base):
    """
    Implements the counting semiring (+, *, 0, 1).

    """

    @staticmethod
    def sum(xs, dim=-1):
        return torch.sum(xs, dim=dim)


class LogSemiring(_BaseLog):
    """
    Implements the log-space semiring (logsumexp, +, -inf, 0).

    Gradients give marginals.
    """

    @staticmethod
    def sum(xs, dim=-1):
        return torch.logsumexp(xs, dim=dim)


class MaxSemiring(_BaseLog):
    """
    Implements the max semiring (max, +, -inf, 0).

    Gradients give argmax.
    """

    @staticmethod
    def sum(xs, dim=-1):
        return torch.max(xs, dim=dim)[0]

    @staticmethod
    def sparse_sum(xs, dim=-1):
        m, a = torch.max(xs, dim=dim)
        return m, (torch.zeros(a.shape).long(), a)


def KMaxSemiring(k):
    """
    Implements the k-max semiring (kmax, +, [-inf, -inf..], [0, -inf, ...]).

    Gradients give k-argmax.
    """

    class KMaxSemiring(_BaseLog):
        @staticmethod
        def size():
            return k

        @classmethod
        def convert(cls, orig_potentials):
            potentials = torch.zeros(
                (k,) + orig_potentials.shape,
                dtype=orig_potentials.dtype,
                device=orig_potentials.device,
            )
            cls.zero_(potentials)
            potentials[0] = orig_potentials
            return potentials

        @classmethod
        def one_(cls, xs):
            cls.zero_(xs)
            xs[0].fill_(0)
            return xs

        @staticmethod
        def unconvert(potentials):
            return potentials[0]

        @staticmethod
        def sum(xs, dim=-1):
            if dim == -1:
                xs = xs.permute(tuple(range(1, xs.dim())) + (0,))
                xs = xs.contiguous().view(xs.shape[:-2] + (-1,))
                xs = torch.topk(xs, k, dim=-1)[0]
                xs = xs.permute((xs.dim() - 1,) + tuple(range(0, xs.dim() - 1)))
                assert xs.shape[0] == k
                return xs
            assert False

        @staticmethod
        def sparse_sum(xs, dim=-1):
            if dim == -1:
                xs = xs.permute(tuple(range(1, xs.dim())) + (0,))
                xs = xs.contiguous().view(xs.shape[:-2] + (-1,))
                xs, xs2 = torch.topk(xs, k, dim=-1)
                xs = xs.permute((xs.dim() - 1,) + tuple(range(0, xs.dim() - 1)))
                xs2 = xs2.permute((xs.dim() - 1,) + tuple(range(0, xs.dim() - 1)))
                assert xs.shape[0] == k
                return xs, (xs2 % k, xs2 // k)
            assert False

        @staticmethod
        def mul(a, b):
            a = a.view((k, 1) + a.shape[1:])
            b = b.view((1, k) + b.shape[1:])
            c = a + b
            c = c.contiguous().view((k * k,) + c.shape[2:])
            ret = torch.topk(c, k, 0)[0]
            assert ret.shape[0] == k
            return ret

    return KMaxSemiring


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
                draws = torch.multinomial(
                    ls.softmax(-1).view(-1, pre_shape[-1]), 1, True
                )
                draws.squeeze(1)
                return (
                    torch.nn.functional.one_hot(draws, pre_shape[-1])
                    .view(*pre_shape)
                    .type_as(ls)
                )

            if dim == -1:
                s = sample(logits)
            else:
                dim = dim if dim >= 0 else logits.dim() + dim
                perm = [i for i in range(logits.dim()) if i != dim] + [dim]
                rev_perm = [a for a, b in sorted(enumerate(perm), key=lambda a: a[1])]
                s = sample(logits.permute(perm)).permute(rev_perm)

            grad_input = grad_output.unsqueeze(dim).mul(s)
        return grad_input, None


class SampledSemiring(_BaseLog):
    """
    Implements a sampling semiring (logsumexp, +, -inf, 0).

    "Gradients" give sample.

    This is an exact forward-filtering, backward-sampling approach.
    """

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

        logits, part, dim = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:

            def sample(ls):
                pre_shape = ls.shape
                draws = torch.multinomial(
                    ls.softmax(-1).view(-1, pre_shape[-1]), 16, True
                )
                draws = draws.transpose(0, 1)
                return (
                    torch.nn.functional.one_hot(draws, pre_shape[-1])
                    .view(16, *pre_shape)
                    .type_as(ls)
                )

            if dim == -1:
                s = sample(logits)
            else:
                dim = dim if dim >= 0 else logits.dim() + dim
                perm = [i for i in range(logits.dim()) if i != dim] + [dim]
                rev_perm = [0] + [
                    a + 1 for a, b in sorted(enumerate(perm), key=lambda a: a[1])
                ]
                s = sample(logits.permute(perm)).permute(rev_perm)

            dim = dim if dim >= 0 else logits.dim() + dim
            final = (grad_output % 2).unsqueeze(0)
            mbits = bits[:].type_as(grad_output)
            on = grad_output.unsqueeze(0) % mbits.view(17, *[1] * grad_output.dim())
            on = on[1:] - on[:-1]
            old_bits = (on + final == 0).unsqueeze(dim + 1)

            grad_input = (
                mbits[:-1]
                .view(16, *[1] * (s.dim() - 1))
                .mul(s.masked_fill_(old_bits, 0))
            )

        return torch.sum(grad_input, dim=0), None


class MultiSampledSemiring(_BaseLog):
    """
    Implements a multi-sampling semiring (logsumexp, +, -inf, 0).

    "Gradients" give up to 16 samples with replacement.
    """

    @staticmethod
    def sum(xs, dim=-1):
        return _MultiSampledLogSumExp.apply(xs, dim)

    @staticmethod
    def to_discrete(xs, j):
        i = j
        final = xs % 2
        mbits = bits.type_as(xs)
        return (((xs % mbits[i + 1]) - (xs % mbits[i]) + final) != 0).type_as(xs)


class EntropySemiring(Semiring):
    """
    Implements an entropy expectation semiring.

    Computes both the log-values and the running distributional entropy.

    Based on descriptions in:

    * Parameter estimation for probabilistic finite-state transducers :cite:`eisner2002parameter`
    * First-and second-order expectation semirings with applications to minimum-risk training on translation forests :cite:`li2009first`
    """

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
        part = torch.logsumexp(xs[0], dim=d)
        log_sm = xs[0] - part.unsqueeze(d)
        sm = log_sm.exp()
        return torch.stack((part, torch.sum(xs[1].mul(sm) - log_sm.mul(sm), dim=d)))

    @staticmethod
    def mul(a, b):
        return torch.stack((a[0] + b[0], a[1] + b[1]))

    @classmethod
    def prod(cls, xs, dim=-1):
        return xs.sum(dim)

    @staticmethod
    def zero_(xs):
        xs[0].fill_(-1e5)
        xs[1].fill_(0)
        return xs

    @staticmethod
    def one_(xs):
        xs[0].fill_(0)
        xs[1].fill_(0)
        return xs


class SparseMaxSemiring(_BaseLog):
    """

    Implements differentiable dynamic programming with a sparsemax semiring (sparsemax, +, -inf, 0).

    Sparse-max gradients give a more sparse set of marginal like terms.

    * From softmax to sparsemax- A sparse model of attention and multi-label classification :cite:`martins2016softmax`
    * Differentiable dynamic programming for structured prediction and attention :cite:`mensch2018differentiable`
    """

    @staticmethod
    def sum(xs, dim=-1):
        return _SimplexProject.apply(xs, dim)


class _SimplexProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, z=1):
        w_star = project_simplex(input, dim)
        ctx.save_for_backward(input, w_star.clone(), torch.tensor(dim))
        x = input.mul(w_star).sum(dim) - w_star.norm(p=2, dim=dim)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        input, w_star, dim = ctx.saved_tensors
        w_star.requires_grad_(True)

        grad_input = None
        if ctx.needs_input_grad[0]:
            wstar = _SparseMaxGrad.apply(w_star, dim)
            grad_input = grad_output.unsqueeze(dim).mul(wstar)
        return grad_input, None, None


class _SparseMaxGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w_star, dim):
        ctx.save_for_backward(w_star, dim)
        return w_star

    @staticmethod
    def backward(ctx, grad_output):
        w_star, dim = ctx.saved_tensors
        return sparsemax_grad(grad_output, w_star, dim.item()), None


def project_simplex(v, dim, z=1):
    v_sorted, _ = torch.sort(v, dim=dim, descending=True)
    cssv = torch.cumsum(v_sorted, dim=dim) - z
    ind = torch.arange(1, 1 + v.shape[dim]).to(dtype=v.dtype)
    cond = v_sorted - cssv / ind >= 0
    k = cond.sum(dim=dim, keepdim=True)
    tau = cssv.gather(dim, k - 1) / k.to(dtype=v.dtype)
    w = torch.clamp(v - tau, min=0)
    return w


def sparsemax_grad(dout, w_star, dim):
    out = dout.clone()
    supp = w_star > 0
    out[w_star <= 0] = 0
    nnz = supp.to(dtype=dout.dtype).sum(dim=dim, keepdim=True)
    out = out - (out.sum(dim=dim, keepdim=True) / nnz)
    out[w_star <= 0] = 0
    return out
