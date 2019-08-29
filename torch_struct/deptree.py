import torch
from .semirings import LogSemiring
from .helpers import _make_chart
import itertools


def _convert(logits):
    "move root arcs from diagonal"
    new_logits = torch.zeros(
        logits.size(0), logits.size(1) + 1, logits.size(2) + 1
    ).type_as(logits.data)
    new_logits.fill_(-1e9)
    new_logits[:, 1:, 1:] = logits
    for i in range(0, logits.size(1)):
        new_logits[:, 0, i + 1] = logits[:, i, i]
        new_logits[:, i + 1, i + 1] = -1e9
    return new_logits


def _unconvert(logits):
    "Move root arcs to diagonal"
    new_logits = torch.zeros(
        logits.size(0), logits.size(1) - 1, logits.size(2) - 1
    ).type_as(logits.data)
    new_logits.fill_(-1e9)
    new_logits[:, :, :] = logits[:, 1:, 1:]
    for i in range(0, new_logits.size(1)):
        new_logits[:, i, i] = logits[:, 0, i + 1]

    return new_logits


# Constants
A, B, R, C, L, I = 0, 1, 1, 1, 0, 0


def deptree_inside(arc_scores, semiring=LogSemiring):
    """
    Compute the inside pass of a projective dependency CRF.

    Parameters:
         arc_scores : b x N x N arc scores with root scores on diagonal.
         semiring

    Returns:
         v: b tensor of total sum
         arcs: list of N,  LR x b x N table

    """
    arc_scores = _convert(arc_scores)
    batch_size, N, _ = arc_scores.shape
    DIRS = 2
    stack = lambda a, b: torch.stack([a, b])
    sstack = lambda a: torch.stack([a, a])

    alpha = [
        [_make_chart((DIRS, batch_size, N, N), arc_scores, semiring) for _ in [I, C]]
        for _ in range(2)
    ]
    arcs = [_make_chart((DIRS, batch_size, N), arc_scores, semiring) for _ in range(N)]

    # Inside step. assumes first token is root symbol
    alpha[A][C][:, :, :, 0].data.fill_(semiring.one())
    alpha[B][C][:, :, :, -1].data.fill_(semiring.one())

    for k in range(1, N):
        f = torch.arange(N - k), torch.arange(k, N)
        arcs[k] = semiring.dot(
            sstack(alpha[A][C][R, :, : N - k, :k]),
            sstack(alpha[B][C][L, :, k:, N - k :]),
            stack(arc_scores[:, f[1], f[0]], arc_scores[:, f[0], f[1]]).unsqueeze(-1),
        )
        alpha[A][I][:, :, : N - k, k] = arcs[k]
        alpha[B][I][:, :, k:N, N - k - 1] = alpha[A][I][:, :, : N - k, k]
        alpha[A][C][:, :, : N - k, k] = semiring.dot(
            stack(
                alpha[A][C][L, :, : N - k, :k], alpha[A][I][R, :, : N - k, 1 : k + 1]
            ),
            stack(
                alpha[B][I][L, :, k:, N - k - 1 : N - 1], alpha[B][C][R, :, k:, N - k :]
            ),
        )
        alpha[B][C][:, :, k:N, N - k - 1] = alpha[A][C][:, :, : N - k, k]
    return alpha[A][C][R, :, 0, N - 1], arcs


def deptree(arc_scores, semiring=LogSemiring):
    """
    Compute the marginals of a projective dependency CRF.

    Parameters:
         arc_scores : b x N x N arc scores with root scores on diagonal.
         semiring

    Returns:
         arc_marginals : b x N x N.

    """
    batch_size, N, _ = arc_scores.shape
    N = N + 1
    v, arcs = deptree_inside(arc_scores, semiring)
    grads = torch.autograd.grad(
        v.sum(dim=0), arcs[1:], create_graph=True, only_inputs=True, allow_unused=False
    )
    ret = torch.zeros(batch_size, N, N).cpu()
    for k, grad in enumerate(grads, 1):
        f = torch.arange(N - k), torch.arange(k, N)
        ret[:, f[0], f[1]] = grad[R].cpu()
        ret[:, f[1], f[0]] = grad[L].cpu()
    return _unconvert(ret)


def deptree_nonproj(arc_scores, eps=1e-5):
    """
    Compute the marginals of a non-projective dependency tree using the
    matrix-tree theorem.

    Allows for overlapping arcs.

    Much faster, but cannot provide a semiring.

    Parameters:
         arc_scores : b x N x N arc scores with root scores on diagonal.
         semiring

    Returns:
         arc_marginals : b x N x N.
    """

    input = arc_scores
    eye = torch.eye(input.shape[1], device=input.device)
    laplacian = input.exp() + eps
    lap = laplacian.masked_fill(eye != 0, 0)
    lap = -lap + torch.diag_embed(lap.sum(1), offset=0, dim1=-2, dim2=-1)
    lap[:, 0] = torch.diagonal(input, 0, -2, -1).exp()
    inv_laplacian = lap.inverse()
    factor = (
        torch.diagonal(inv_laplacian, 0, -2, -1)
        .unsqueeze(2)
        .expand_as(input)
        .transpose(1, 2)
    )
    term1 = input.exp().mul(factor).clone()
    term2 = input.exp().mul(inv_laplacian.transpose(1, 2)).clone()
    term1[:, :, 0] = 0
    term2[:, 0] = 0
    output = term1 - term2
    roots_output = (
        torch.diagonal(input, 0, -2, -1).exp().mul(inv_laplacian.transpose(1, 2)[:, 0])
    )
    output = output + torch.diag_embed(roots_output, 0, -2, -1)
    return output


### Tests


def deptree_check(arc_scores, semiring=LogSemiring, non_proj=False):
    parses = []
    q = []
    arc_scores = _convert(arc_scores)
    batch_size, N, _ = arc_scores.shape
    for mid in itertools.product(range(N + 1), repeat=N - 1):
        parse = [-1] + list(mid)
        if not _is_spanning(parse):
            continue
        if not non_proj and not _is_projective(parse):
            continue
        q.append(parse)
        parses.append(
            semiring.times(*[arc_scores[0][parse[i], i] for i in range(1, N, 1)])
        )
    return semiring.sum(torch.tensor(parses))


def _is_spanning(parse):
    """
    Is the parse tree a valid spanning tree?
    Returns
    --------
    spanning : bool
    True if a valid spanning tree.
    """
    d = {}
    for m, h in enumerate(parse):
        if m == h:
            return False
        d.setdefault(h, [])
        d[h].append(m)
    stack = [0]
    seen = set()
    while stack:
        cur = stack[0]
        if cur in seen:
            return False
        seen.add(cur)
        stack = d.get(cur, []) + stack[1:]
    if len(seen) != len(parse) - len([1 for p in parse if p is None]):
        return False
    return True


def _is_projective(parse):
    """
    Is the parse tree projective?
    Returns
    --------
    projective : bool
       True if a projective tree.
    """
    for m, h in enumerate(parse):
        for m2, h2 in enumerate(parse):
            if m2 == m:
                continue
            if m < h:
                if (
                    m < m2 < h < h2
                    or m < h2 < h < m2
                    or m2 < m < h2 < h
                    or h2 < m < m2 < h
                ):
                    return False
            if h < m:
                if (
                    h < m2 < m < h2
                    or h < h2 < m < m2
                    or m2 < h < h2 < m
                    or h2 < h < m2 < m
                ):
                    return False
    return True
