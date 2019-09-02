import torch
from .semirings import LogSemiring
from .helpers import _make_chart


def semimarkov_forward(edge, semiring=LogSemiring, lengths=None, force_grad=False):
    """
    Compute the forward pass of a semimarkov CRF.

    Parameters:
         edge : b x N x K x C x C semimarkov potentials
         semiring
         lengths: None or b long tensor mask

    Returns:
         v: b tensor of total sum
         spans: list of N,  b x K x C x C table

    """
    batch, N, K, C, C2 = edge.shape
    if lengths is None:
        lengths = torch.LongTensor([N] * batch)
    assert max(lengths) <= N, "Length longer than edge scores"
    assert C == C2, "Transition shape doesn't match"

    spans = [None for _ in range(N)]
    alpha = [
        _make_chart((batch, K, C), edge, semiring, force_grad) for n in range(N + 1)
    ]
    beta = [_make_chart((batch, C), edge, semiring, force_grad) for n in range(N + 1)]
    beta[0].data.fill_(semiring.one())
    for n in range(1, N + 1):
        spans[n - 1] = semiring.times(
            beta[n - 1].view(batch, 1, 1, C), edge[:, n - 1].view(batch, K, C, C)
        )
        alpha[n - 1][:] = semiring.sum(spans[n - 1])
        t = max(n - K, -1)
        f1 = torch.arange(n - 1, t, -1)
        f2 = torch.arange(1, len(f1) + 1)
        print(n - 1, f1, f2)
        beta[n][:] = semiring.sum(
            torch.stack([alpha[a][:, b] for a, b in zip(f1, f2)]), dim=0
        )
    v = semiring.sum(torch.stack([beta[l][i] for i, l in enumerate(lengths)]), dim=1)
    return v, spans


def semimarkov(edge, semiring=LogSemiring, lengths=None):
    """
    Compute the marginals of a semimarkov CRF.

    Parameters:
         edge : b x N x K x C x C semimarkov potentials
         semiring

    Returns:
         marginals: b x N x K x C table

    """
    v, spans = semimarkov_forward(edge, semiring, lengths, force_grad=True)
    marg = torch.autograd.grad(
        v.sum(dim=0), spans, create_graph=True, only_inputs=True, allow_unused=False
    )
    return torch.stack(marg, dim=1)


# Tests


def semimarkov_check(edge, semiring=LogSemiring):
    batch, N, K, C, _ = edge.shape
    chains = {}
    chains[0] = [([(c, 0)], torch.zeros(batch).fill_(semiring.one())) for c in range(C)]

    for n in range(1, N + 1):
        chains[n] = []
        for k in range(1, K):
            if n - k not in chains:
                continue
            for chain, score in chains[n - k]:
                for c in range(C):
                    chains[n].append(
                        (
                            chain + [(c, k)],
                            semiring.mul(score, edge[:, n - k, k, c, chain[-1][0]]),
                        )
                    )
    return semiring.sum(torch.stack([s for (_, s) in chains[N]]), dim=0)
