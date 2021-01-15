import torch
import torch.distributions
from .semirings import _BaseLog


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


def GumbelMaxSemiring(temp):
    class _GumbelMaxLogSumExp(torch.autograd.Function):
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
                    update = (
                        ls + torch.distributions.Gumbel(0, 1).sample((ls.shape[-1],))
                    ) / temp
                    out = torch.nn.functional.one_hot(update.max(-1)[1], pre_shape[-1])
                    return out

                if dim == -1:
                    s = sample(logits)
                else:
                    dim = dim if dim >= 0 else logits.dim() + dim
                    perm = [i for i in range(logits.dim()) if i != dim] + [dim]
                    rev_perm = [
                        a for a, b in sorted(enumerate(perm), key=lambda a: a[1])
                    ]
                    s = sample(logits.permute(perm)).permute(rev_perm)

                grad_input = grad_output.unsqueeze(dim).mul(s)
            return grad_input, None

    class _GumbelMaxSemiring(_BaseLog):
        @staticmethod
        def sum(xs, dim=-1):
            return _GumbelMaxLogSumExp.apply(xs, dim)

    return _GumbelMaxSemiring


def GumbelCRFSemiring(temp):
    class ST(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, dim):
            out = torch.nn.functional.one_hot(logits.max(-1)[1], dim)
            out = out.type_as(logits)
            ctx.save_for_backward(logits, out)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            logits, out = ctx.saved_tensors
            with torch.enable_grad():
                ret = torch.autograd.grad(
                    logits.softmax(-1), logits, out * grad_output
                )[0]
            return ret, None

    class _GumbelCRFLogSumExp(torch.autograd.Function):
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
                    update = (
                        ls + torch.distributions.Gumbel(0, 1).sample((ls.shape[-1],))
                    ) / temp
                    out = ST.apply(update, ls.shape[-1])
                    return out

                if dim == -1:
                    s = sample(logits)
                else:
                    dim = dim if dim >= 0 else logits.dim() + dim
                    perm = [i for i in range(logits.dim()) if i != dim] + [dim]
                    rev_perm = [
                        a for a, b in sorted(enumerate(perm), key=lambda a: a[1])
                    ]
                    s = sample(logits.permute(perm)).permute(rev_perm)

                grad_input = grad_output.unsqueeze(dim).mul(s)
            return grad_input, None

    class _GumbelCRFSemiring(_BaseLog):
        @staticmethod
        def sum(xs, dim=-1):
            return _GumbelCRFLogSumExp.apply(xs, dim)

    return _GumbelCRFSemiring


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
