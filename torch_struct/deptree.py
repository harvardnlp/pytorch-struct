import torch
import itertools
from .helpers import _Struct


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


class DepTree(_Struct):
    def sum(self, arc_scores, lengths=None):
        """
        Compute the inside pass of a projective dependency CRF.

        Parameters:
            arc_scores : b x N x N arc scores with root scores on diagonal.
            semiring
            lengths: None or b long tensor mask

        Returns:
            v: b tensor of total sum
            arcs: list of N,  LR x b x N table

        """
        return self._dp(arc_scores, lengths)[0]

    def _dp(self, arc_scores, lengths=None, force_grad=False):
        semiring = self.semiring
        arc_scores = _convert(arc_scores)
        batch, N, N2 = arc_scores.shape

        assert N == N2, "Non-square potentials"
        DIRS = 2
        if lengths is None:
            lengths = torch.LongTensor([N - 1] * batch)
        assert max(lengths) <= N, "Length longer than N"
        for b in range(batch):
            arc_scores[b, lengths[b] + 1 :, :] = semiring.zero()
            arc_scores[b, :, lengths[b] + 1 :] = semiring.zero()

        def stack(a, b):
            return torch.stack([a, b])

        def sstack(a):
            return torch.stack([a, a])

        alpha = [
            self._make_chart(2, (DIRS, batch, N, N), arc_scores, force_grad)
            for _ in range(2)
        ]
        arcs = self._make_chart(N, (DIRS, batch, N), arc_scores, force_grad)

        # Inside step. assumes first token is root symbol
        alpha[A][C][:, :, :, 0].data.fill_(semiring.one())
        alpha[B][C][:, :, :, -1].data.fill_(semiring.one())

        for k in range(1, N):
            f = torch.arange(N - k), torch.arange(k, N)
            arcs[k] = semiring.dot(
                sstack(alpha[A][C][R, :, : N - k, :k]),
                sstack(alpha[B][C][L, :, k:, N - k :]),
                stack(arc_scores[:, f[1], f[0]], arc_scores[:, f[0], f[1]]).unsqueeze(
                    -1
                ),
            )
            alpha[A][I][:, :, : N - k, k] = arcs[k]
            alpha[B][I][:, :, k:N, N - k - 1] = alpha[A][I][:, :, : N - k, k]
            alpha[A][C][:, :, : N - k, k] = semiring.dot(
                stack(
                    alpha[A][C][L, :, : N - k, :k],
                    alpha[A][I][R, :, : N - k, 1 : k + 1],
                ),
                stack(
                    alpha[B][I][L, :, k:, N - k - 1 : N - 1],
                    alpha[B][C][R, :, k:, N - k :],
                ),
            )
            alpha[B][C][:, :, k:N, N - k - 1] = alpha[A][C][:, :, : N - k, k]
        return (
            torch.stack([alpha[A][C][R, i, 0, l] for i, l in enumerate(lengths)]),
            arcs,
        )

    def marginals(self, arc_scores, lengths=None):
        """
        Compute the marginals of a projective dependency CRF.

        Parameters:
            arc_scores : b x N x N arc scores with root scores on diagonal.
            semiring
            lengths
        Returns:
            arc_marginals : b x N x N.

        """
        batch, N, _ = arc_scores.shape
        N = N + 1
        v, arcs = self._dp(arc_scores, lengths, force_grad=True)
        grads = torch.autograd.grad(
            v.sum(dim=0),
            arcs[1:],
            create_graph=True,
            only_inputs=True,
            allow_unused=False,
        )
        ret = torch.zeros(batch, N, N).cpu()
        for k, grad in enumerate(grads, 1):
            f = torch.arange(N - k), torch.arange(k, N)
            ret[:, f[0], f[1]] = grad[R].cpu()
            ret[:, f[1], f[0]] = grad[L].cpu()
        return _unconvert(ret)

    @staticmethod
    def to_parts(sequence, extra=None, lengths=None):
        """
        Convert a sequence representation to arcs

        Parameters:
            sequence : b x N long tensor in [0, N] (indexing is +1)
        Returns:
            arcs : b x N x N arc indicators
        """
        batch, N = sequence.shape
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)
        labels = torch.zeros(batch, N + 1, N + 1).long()
        for n in range(1, N + 1):
            labels[torch.arange(batch), sequence[:, n - 1], n] = 1
        for b in range(batch):
            labels[b, lengths[b] + 1 :, :] = 0
            labels[b, :, lengths[b] + 1 :] = 0
        return _unconvert(labels)

    @staticmethod
    def from_parts(arcs):
        """
        Convert a arc representation to sequence

        Parameters:
            arcs : b x N x N arc indicators
        Returns:
            sequence : b x N long tensor in [0, N] (indexing is +1)
        """
        batch, N, _ = arcs.shape
        labels = torch.zeros(batch, N).long()
        on = arcs.nonzero()
        for i in range(on.shape[0]):
            if on[i][1] == on[i][2]:
                labels[on[i][0], on[i][2]] = 0
            else:
                labels[on[i][0], on[i][2]] = on[i][1] + 1
        return labels, None

    @staticmethod
    def _rand():
        b = torch.randint(2, 4, (1,))
        N = torch.randint(2, 4, (1,))
        return torch.rand(b, N, N), (b.item(), N.item())

    def enumerate(self, arc_scores, non_proj=False):
        semiring = self.semiring
        parses = []
        q = []
        arc_scores = _convert(arc_scores)
        batch, N, _ = arc_scores.shape
        for mid in itertools.product(range(N + 1), repeat=N - 1):
            parse = [-1] + list(mid)
            if not _is_spanning(parse):
                continue
            if not non_proj and not _is_projective(parse):
                continue
            q.append(parse)
            parses.append(
                semiring.times(*[arc_scores[:, parse[i], i] for i in range(1, N, 1)])
            )
        return semiring.sum(torch.stack(parses, dim=-1))


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
