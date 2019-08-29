import torch
from .semirings import LogSemiring
from .helpers import _make_chart

A, B = 0, 1


def cky_inside(terms, rules, roots, semiring=LogSemiring):
    """
    Compute the inside pass of a CFG using CKY.

    Parameters:
         terms : b x n x T
         rules : b x NT x (NT+T) x (NT+T)
         root:   b x NT
         semiring

    Returns:
         v: b tensor of total sum
         spans: list of N,  b x N x (NT+t)
    """
    batch_size, N, T = terms.shape
    _, NT, _, _ = rules.shape

    beta = [_make_chart((batch_size, N, N, NT + T), rules, semiring) for _ in range(2)]

    span = [_make_chart((batch_size, N, NT + T), rules, semiring) for _ in range(N)]

    beta[A][:, :, 0, NT:] = terms
    beta[B][:, :, N - 1, NT:] = terms

    S = NT + T
    for w in range(1, N):
        Y = beta[A][:, : N - w, :w, :].view(batch_size, N - w, w, 1, S, 1)
        Z = beta[B][:, w:, N - w :, :].view(batch_size, N - w, w, 1, 1, S)
        X_Y_Z = rules.view(batch_size, 1, NT, S, S)
        rulesmid = semiring.times(semiring.sum(semiring.times(Y, Z), dim=2), X_Y_Z)
        rulesmid = rulesmid.view(batch_size, N - w, NT, S * S)
        span[w] = semiring.sum(rulesmid, dim=3)
        beta[A][:, : N - w, w, :NT] = span[w]
        beta[B][:, w:N, N - w - 1, :NT] = beta[A][:, : N - w, w, :NT]

    log_Z = semiring.dot(beta[A][:, 0, N - 1, :NT], roots)
    return log_Z, span


def cky(terms, rules, roots, semiring=LogSemiring):
    """
    Compute the marginals of a CFG using CKY.

    Parameters:
         terms : b x n x T
         rules : b x NT x (NT+T) x (NT+T)
         root:   b x NT
         semiring

    Returns:
         v: b tensor of total sum
         spans: b x N x N x (NT+t) span marginals
                where spans[:, i, d] covers (i, i + d)
    """
    v, span = cky_inside(terms, rules, roots, semiring=LogSemiring)
    marg = torch.autograd.grad(
        v.sum(dim=0), span[1:], create_graph=True, only_inputs=True, allow_unused=False
    )
    return torch.stack(marg, dim=1)


###### Test


def cky_check(terms, rules, roots, semiring=LogSemiring):
    batch_size, N, T = terms.shape
    _, NT, _, _ = rules.shape

    def enumerate(x, start, end):
        if start + 1 == end:
            yield (terms[0, start, x - NT], [(start, x - NT)])
        else:
            for w in range(start + 1, end):
                for y in range(NT) if w != start + 1 else range(NT, NT + T):
                    for z in range(NT) if w != end - 1 else range(NT, NT + T):
                        children = []
                        for m1, y1 in enumerate(y, start, w):
                            for m2, z1 in enumerate(z, w, end):
                                yield (
                                    semiring.times(
                                        semiring.times(m1, m2), rules[0, x, y, z]
                                    ),
                                    [(x, start, w, end)] + y1 + z1,
                                )

    for nt in range(NT):
        print(list(enumerate(nt, 0, N)))
    ls = []
    for nt in range(NT):
        ls += [semiring.times(s, roots[0, nt]) for s, _ in enumerate(nt, 0, N)]
    return semiring.sum(torch.stack(ls))
