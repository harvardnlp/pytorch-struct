import torch
from .semirings import LogSemiring
from torch.autograd import Function


# def roll(a, b, N, k, gap=0):
#     return (a[:, : N - (k + gap), (k + gap) :], b[:, k + gap :, : N - (k + gap)])


# def roll2(a, b, N, k, gap=0):
#     return (a[:, :, : N - (k + gap), (k + gap) :], b[:, :, k + gap :, : N - (k + gap)])


class _Struct:
    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

    def score(self, potentials, parts, batch_dims=[0]):
        score = torch.mul(potentials, parts)
        batch = tuple((score.shape[b] for b in batch_dims))
        return self.semiring.prod(score.view(batch + (-1,)))

    def _make_chart(self, N, size, potentials, force_grad=False):
        return [
            (
                self.semiring.zero_(
                    torch.zeros(
                        *((self.semiring.size(),) + size),
                        dtype=potentials.dtype,
                        device=potentials.device
                    )
                ).requires_grad_(force_grad and not potentials.requires_grad)
            )
            for _ in range(N)
        ]

    def sum(self, edge, lengths=None, _autograd=True, _raw=False):
        """
        Compute the (semiring) sum over all structures model.

        Parameters:
            params : generic params (see class)
            lengths: None or b long tensor mask

        Returns:
            v: b tensor of total sum
        """

        if (
            _autograd
            or self.semiring is not LogSemiring
            or not hasattr(self, "_dp_backward")
        ):

            v = self._dp(edge, lengths)[0]
            if _raw:
                return v
            return self.semiring.unconvert(v)

        else:
            v, _, alpha = self._dp(edge, lengths, False)

            class DPManual(Function):
                @staticmethod
                def forward(ctx, input):
                    return v

                @staticmethod
                def backward(ctx, grad_v):
                    marginals = self._dp_backward(edge, lengths, alpha)
                    return marginals.mul(
                        grad_v.view((grad_v.shape[0],) + tuple([1] * marginals.dim()))
                    )

            return DPManual.apply(edge)

    def marginals(self, edge, lengths=None, _autograd=True, _raw=False):
        """
        Compute the marginals of a structured model.

        Parameters:
            params : generic params (see class)
            lengths: None or b long tensor mask
        Returns:
            marginals: b x (N-1) x C x C table

        """
        if (
            _autograd
            or self.semiring is not LogSemiring
            or not hasattr(self, "_dp_backward")
        ):
            v, edges, _ = self._dp(edge, lengths=lengths, force_grad=True)
            if _raw:
                all_m = []
                print(v)
                for k in range(v.shape[0]):
                    obj = v[k].sum(dim=0)

                    marg = torch.autograd.grad(
                        obj,
                        edges,
                        create_graph=True,
                        only_inputs=True,
                        allow_unused=False,
                    )
                    all_m.append(self.semiring.unconvert(
                        self._arrange_marginals(marg)))
                return torch.stack(all_m, dim=0)
            else:
                obj = self.semiring.unconvert(v).sum(dim=0)
                marg = torch.autograd.grad(
                    obj,
                    edges,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=False,
                )
                a_m = self._arrange_marginals(marg)
                return self.semiring.unconvert(a_m)
        else:
            v, _, alpha = self._dp(edge, lengths=lengths, force_grad=True)
            return self._dp_backward(edge, lengths, alpha)

    @staticmethod
    def to_parts(spans, extra, lengths=None):
        return spans

    @staticmethod
    def from_parts(spans):
        return spans, None

    def _arrange_marginals(self, marg):
        return marg[0]
