import torch
from .semirings import LogSemiring
from torch.autograd import Function

class DPManual(Function):
    @staticmethod
    def forward(ctx, obj, input, lengths):
        v, _, alpha = obj._dp(input, lengths, False)
        ctx.obj = obj
        ctx.lengths = lengths
        ctx.alpha = alpha

        if isinstance(input, tuple):
            ctx.save_for_backward(*input)
        else:
            ctx.save_for_backward(input)
        return v

    @staticmethod
    def backward(ctx, grad_v):
        input = ctx.saved_tensors
        if len(input) == 1:
            input = input[0]
        marginals = ctx.obj._dp_backward(input, ctx.lengths, ctx.alpha)
        return None, marginals, None


class _Struct:
    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

    def score(self, potentials, parts):
        batch = potentials.shape[0]
        return torch.mul(potentials, parts).view(batch, -1).sum(-1)

    def _make_chart(self, N, size, potentials, force_grad):
        return [
            (
                torch.zeros(*size)
                .type_as(potentials)
                .fill_(self.semiring.zero())
                .requires_grad_(force_grad and not potentials.requires_grad)
            )
            for _ in range(N)
        ]
