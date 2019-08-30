import torch
from .semirings import LogSemiring
from .helpers import _make_chart


def linearchain_forward(edge, semiring=LogSemiring):
    """
    Compute the forward pass of a linear chain CRF.

    Parameters:
         edge : b x N x C x C markov potentials
                    (n-1 x z_n x z_{n-1})
         semiring

    Returns:
         v: b tensor of total sum
         inside: list of N,  b x C x C table

    """
    batch, N, C, _ = edge.shape
    alpha = [_make_chart((batch, C), edge, semiring) for n in range(N + 1)]
    edge_store = [None for _ in range(N)]
    alpha[0].data.fill_(semiring.one())
    for n in range(1, N + 1):
        edge_store[n - 1] = semiring.times(
            alpha[n - 1].view(batch, 1, C), edge[:, n - 1]
        )
        alpha[n] = semiring.sum(edge_store[n - 1])
    return semiring.sum(alpha[N]), edge_store


def linearchain(edge, semiring=LogSemiring):
    """
    Compute the marginals of a linear chain CRF.

    Parameters:
         edge : b x N x C x C markov potentials
                    (t x z_t x z_{t-1})
         semiring

    Returns:
         marginals: b x N x C x C table

    """
    v, alpha = linearchain_forward(edge, semiring)
    marg = torch.autograd.grad(
        v.sum(dim=0), alpha, create_graph=True, only_inputs=True, allow_unused=False
    )
    return torch.stack(marg, dim=1)


# Adapters
def hmm(transition, emission, init, observations):
    """
    Convert HMM to a linear chain.

    Parameters:
        transition: C X C
        emission: V x C
        init: C
        observations: b x N between [0, V-1]

    Returns:
        edges: b x N x C x C
    """
    V, C = emission.shape
    batch, N = observations.shape
    scores = torch.ones(batch, N, C, C).type_as(emission)
    scores[:, :, :, :] *= transition.view(1, 1, C, C)
    scores[:, 0, :, :] *= init.view(1, 1, C)
    obs = emission[observations.view(batch * N), :]
    scores[:, :, :, :] *= obs.view(batch, N, 1, C)
    return scores


### Tests
def linearchain_check(edge, semiring=LogSemiring):
    batch, N, C, _ = edge.shape
    chains = [([c], torch.zeros(batch).fill_(semiring.one())) for c in range(C)]
    for n in range(1, N + 1):
        new_chains = []
        for chain, score in chains:
            for c in range(C):
                new_chains.append(
                    (chain + [c], semiring.mul(score, edge[:, n - 1, c, chain[-1]]))
                )
        chains = new_chains

    return semiring.sum(torch.stack([s for (_, s) in chains]), dim=0)
