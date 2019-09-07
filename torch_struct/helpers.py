import torch
from .semirings import LogSemiring
from torch.autograd import Function


def roll(a, b, N, k, gap=0):
    return (a[:, :N - (k+gap), (k+gap):], \
            b[:,  k+gap:, : N-(k+gap)])




class DPManual(Function):
    @staticmethod
    def forward(ctx, obj, input, lengths):
        with torch.no_grad():
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
        with torch.no_grad():
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


    def sum(self, edge, lengths=None, _autograd=False):
        """
        Compute the (semiring) sum over all structures model.

        Parameters:
            params : generic params (see class)
            lengths: None or b long tensor mask

        Returns:
            v: b tensor of total sum

        """
        if _autograd or not self.semiring is LogSemiring or "_dp_backward" not in self.__dict__:
            return self._dp(edge, lengths)[0]
        else:
            return DPManual.apply(self, edge, lengths)

    def marginals(self, edge, lengths=None, _autograd=False):
        """
        Compute the marginals of a structured model.

        Parameters:
            params : generic params (see class)
            lengths: None or b long tensor mask
        Returns:
            marginals: b x (N-1) x C x C table

        """
        v, edge, alpha = self._dp(edge, lengths=lengths, force_grad=True)
        if _autograd or not self.semiring is LogSemiring or "_dp_backward" not in self.__dict__:
            marg = torch.autograd.grad(
                v.sum(dim=0), edge, create_graph=True, only_inputs=True, allow_unused=False
            )
            return self._arrange_marginals(marg)
        else:
            return self._dp_backward(edge, lengths, alpha)
