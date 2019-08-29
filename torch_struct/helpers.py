import torch


def _make_chart(size, potentials, semiring):
    return (
        torch.zeros(*size)
        .type_as(potentials)
        .fill_(semiring.zero())
        .requires_grad_(True)
    )
