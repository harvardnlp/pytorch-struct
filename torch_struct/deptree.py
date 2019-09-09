import torch
import itertools
from .helpers import _Struct, roll2


def _convert(logits):
    "move root arcs from diagonal"
    new_logits = torch.zeros(
        logits.size(0), logits.size(1) + 1, logits.size(2) + 1
    ).type_as(logits.data)
    new_logits.fill_(-1e9)
    new_logits[:, 1:, 1:] = logits

    N = logits.size(1)
    new_logits[:, 0, 1:] = logits[:, torch.arange(N), torch.arange(N)]
    new_logits[:, torch.arange(1, N), torch.arange(1, N)] = -1e9
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
    """
    A projective dependency CRF.

    Parameters:
    arc_scores : b x N x N arc scores with root scores on diagonal.
    """

    def _dp(self, arc_scores, lengths=None, force_grad=False):
        semiring = self.semiring
        arc_scores = _convert(arc_scores)
        batch, N, lengths = self._check_potentials(arc_scores, lengths)

        DIRS = 2

        alpha = [
            self._make_chart(2, (DIRS, batch, N, N), arc_scores, force_grad)
            for _ in range(2)
        ]

        def stack(a, b):
            return torch.stack([a, b])

        def sstack(a):
            return torch.stack([a, a])

        arcs = [self._make_chart(1, (DIRS, batch, N-k), arc_scores, force_grad)[0]
                for k in range(N)]

        # Inside step. assumes first token is root symbol
        alpha[A][C][:, :, :, 0].data.fill_(semiring.one())
        alpha[B][C][:, :, :, -1].data.fill_(semiring.one())
        k = 0

        AIR = alpha[A][I][R, :, :N-k, 1:k]
        BIL = alpha[B][I][L, :, k:N, N-k:N-1]
        k = 1
        AC2 = alpha[A][C][:, :, :N - k, :k]
        BC2 = alpha[B][C][:, :, k:, N - k:]
        ends = [None]
        for k in range(1, N):
            f = torch.arange(N - k), torch.arange(k, N)
            if k > 1:
                AC2 = torch.cat([AC[:, :, :-1], AC_next[:, :, :-1].unsqueeze(-1)],
                                 dim=3)
            if k > 1:
                BC2 = torch.cat([AC_next[:, :, 1:].unsqueeze(-1), BC[:, :, 1:]], dim=3)

            start = semiring.dot(BC2[L], AC2[R])
            if k == 1:
                arcs[k] = stack(semiring.times(start,   arc_scores[:, f[1], f[0]]),
                                semiring.times(start, arc_scores[:, f[0], f[1]]))

            else:
                arcs[k] = stack(semiring.times(start),   #, arc_scores[:, f[1], f[0]]),
                                semiring.times(start)) #, arc_scores[:, f[0], f[1]]))

            AIR2 = torch.cat([AIR[:, :-1], arcs[k][R].unsqueeze(-1)], dim=2)
            BIL2 = torch.cat([arcs[k][L].unsqueeze(-1), BIL[:, 1:]], dim=2)
            AC_next = stack(semiring.dot(AC2[L], BIL2), semiring.dot(AIR2, BC2[R]))

            ends.append(AC_next[R, :, 0])
            AC = AC2
            BC = BC2
            AIR = AIR2
            BIL = BIL2
        v = torch.stack([ends[l][i] for i, l in enumerate(lengths)])
        # v = torch.stack([alpha[A][C][R, i, 0, l] for i, l in enumerate(lengths)])
        return (v, arcs[1:], alpha)

    def _check_potentials(self, arc_scores, lengths=None):
        semiring = self.semiring
        batch, N, N2 = arc_scores.shape
        assert N == N2, "Non-square potentials"
        if lengths is None:
            lengths = torch.LongTensor([N - 1] * batch)
        assert max(lengths) <= N, "Length longer than N"
        for b in range(batch):
            arc_scores[b, lengths[b] + 1 :, :] = semiring.zero()
            arc_scores[b, :, lengths[b] + 1 :] = semiring.zero()

        return batch, N, lengths

    def _dp_backward(self, arc_scores, lengths, alpha_in, v=None, force_grad=False):

        # This function is super complicated.
        semiring = self.semiring
        arc_scores = _convert(arc_scores)
        batch, N, lengths = self._check_potentials(arc_scores, lengths)
        DIRS = 2

        alpha = [
            self._make_chart(2, (DIRS, batch, N, N), arc_scores, force_grad)
            for _ in range(2)
        ]

        def stack(a, b, dim=-1):
            return torch.stack([a, b], dim=dim)

        def sstack(a):
            return torch.stack([a, a], dim=-1)

        for k in range(N - 1, -1, -1):
            # Initialize
            if N - k - 1 > 0:
                # R completes
                # I -> C* C
                # I -> C* C
                # C -> I C*
                a = semiring.sum(
                    semiring.times(
                        *roll2(
                            stack(
                                stack(alpha[A][I][R], alpha[A][I][L]),
                                sstack(alpha_in[B][C][R]),
                                dim=0,
                            ),
                            stack(
                                sstack(alpha_in[A][C][L]),
                                stack(alpha[B][I][L], alpha[B][I][R]),
                                dim=0,
                            ),
                            N,
                            k,
                            1,
                        )
                    ).view(2, batch, N - k - 1, -1),
                    dim=-1,
                )
                alpha[A][C][L, :, 1 : N - k, k] = a[1]
                alpha[A][C][R, :, : N - k - 1, k] = a[0]

            for b, l in enumerate(lengths):
                if l == k:
                    alpha[A][C][R, b, 0, l] = semiring.one()
                    alpha[B][C][R, b, l, N - l - 1] = semiring.one()

            c = semiring.sum(
                semiring.times(
                    *roll2(
                        stack(alpha[A][C][L], alpha_in[B][I][R], dim=0),
                        stack(alpha_in[A][I][L], alpha[B][C][R], dim=0),
                        N,
                        k,
                        0,
                    )
                )
            )
            alpha[A][C][:, :, : N - k, k] = semiring.plus(
                alpha[A][C][:, :, : N - k, k], c
            )

            # Compute reverses.
            alpha[B][C][:, :, k:N, N - k - 1] = alpha[A][C][:, :, : N - k, k]

            if k > 0:
                f = torch.arange(N - k), torch.arange(k, N)
                alpha[A][I][:, :, : N - k, k] = semiring.dot(
                    stack(
                        arc_scores[:, f[1], f[0]], arc_scores[:, f[0], f[1]], dim=0
                    ).unsqueeze(-1),
                    *roll2(
                        stack(alpha_in[B][C][L], alpha[A][C][R], dim=0),
                        stack(alpha[B][C][L], alpha_in[A][C][R], dim=0),
                        N,
                        k,
                    )
                )
                alpha[B][I][:, :, k:N, N - k - 1] = alpha[A][I][:, :, : N - k, k]

        v = alpha[A][C][R, :, 0, 0]
        left = semiring.times(alpha[A][I][L, :, :, :], alpha_in[A][I][L, :, :, :])
        right = semiring.times(alpha[A][I][R, :, :, :], alpha_in[A][I][R, :, :, :])
        ret = torch.zeros(batch, N, N).type_as(left)
        for k in torch.arange(N):
            f = torch.arange(N - k), torch.arange(k, N)
            ret[:, f[1], k] = left[:, k, f[0]]
            ret[:, k, f[1]] = right[:, k, f[0]]

        ret = semiring.div_exp(ret - arc_scores, v.view(batch, 1, 1))
        return _unconvert(ret)

    def _arrange_marginals(self, grads):
        batch, N = grads[0][0].shape
        N = N + 1
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
