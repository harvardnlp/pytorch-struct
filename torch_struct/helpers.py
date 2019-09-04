import torch
from .semirings import LogSemiring


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
