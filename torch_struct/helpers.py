import torch


def _make_chart(size, potentials, semiring, force_grad):
    return (
        torch.zeros(*size)
        .type_as(potentials)
        .fill_(semiring.zero())
        .requires_grad_(force_grad and not potentials.requires_grad)
    )
