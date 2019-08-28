import torch
from .semirings import LogSemiring
from .helpers import _make_chart

def semimarkov_inside(edge, semiring=LogSemiring):
    batch, N, K, C, _ = edge.shape
    spans = [None for n in range(N+1)]
    alpha = _make_chart((batch, N+1, K, C), edge, semiring)


    beta = [_make_chart((batch, C), edge, semiring)
             for n in range(N+1)]
    beta[0].data.fill_(semiring.one())
    for n in range(1, N + 1):
        spans[n-1] = semiring.dot(beta[n-1].view(batch, 1, 1, C),
                                  edge[:, n-1].view(batch, K, C, C))
        alpha[:, n-1] = spans[n-1]
        # tmp.view(batch, K, C)
        t = max(n - K, -1)
        f1 = torch.arange(n-1, t,-1)
        f2 = torch.arange(1, len(f1)+1)
        beta[n] = semiring.sum(alpha[:, f1, f2], dim=1)

    return semiring.sum(beta[N], dim=1)


def semimarkov(edge, semiring=LogSemiring):
    v = semimarkov_inside(edge, semiring)
    grads = torch.autograd.grad(v.sum(dim=0), alpha, create_graph=True,
                                only_inputs=True, allow_unused=False)
    return grads

def semimarkov_check(edge, semiring=LogSemiring):
    batch, N, K, C, _ = edge.shape
    chains = {}
    chains[0] = [([(c, 0)], torch.zeros(batch).fill_(semiring.one()))
                 for c in range(C)]

    for n in range(1, N + 1):
        chains[n] = []
        for k in range(1, K):
            if n -k not in chains: continue
            for chain, score in chains[n-k]:
                for c in range(C):
                    chains[n].append((chain + [(c, k)],
                                      semiring.mul(score,
                                                   edge[:, n-k, k, c, chain[-1][0]])))
    print(chains[N])
    return semiring.sum(torch.stack([s for (_, s) in chains[N]]), dim=0)
