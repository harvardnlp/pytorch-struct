import torch_struct
import torch
from torch_struct import LogSemiring
import itertools


class LinearChainTest:
    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

    @staticmethod
    def _rand(min_n=2):
        b = torch.randint(2, 4, (1,))
        N = torch.randint(min_n, 4, (1,))
        C = torch.randint(2, 4, (1,))
        return torch.rand(b, N, C, C), (b.item(), (N + 1).item())

    ### Tests

    def enumerate(self, edge, lengths=None):
        model = torch_struct.LinearChain(self.semiring)
        semiring = self.semiring
        ssize = semiring.size()
        edge, batch, N, C, lengths = model._check_potentials(edge, lengths)
        chains = [[([c], semiring.one_(torch.zeros(ssize, batch))) for c in range(C)]]

        enum_lengths = torch.LongTensor(lengths.shape)
        for n in range(1, N):
            new_chains = []
            for chain, score in chains[-1]:
                for c in range(C):
                    new_chains.append(
                        (
                            chain + [c],
                            semiring.mul(score, edge[:, :, n - 1, c, chain[-1]]),
                        )
                    )
            chains.append(new_chains)

            for b in range(lengths.shape[0]):
                if lengths[b] == n + 1:
                    enum_lengths[b] = len(new_chains)

        edges = model.to_parts(
            torch.stack([torch.tensor(c) for (c, _) in chains[-1]]), C
        )
        # Sum out non-batch
        a = torch.einsum("ancd,sbncd->sbancd", edges.float(), edge)
        a = semiring.prod(a.view(*a.shape[:3] + (-1,)), dim=3)
        a = semiring.sum(a, dim=2)
        ret = semiring.sum(torch.stack([s for (_, s) in chains[-1]], dim=1), dim=1)
        assert torch.isclose(a, ret).all(), "%s %s" % (a, ret)

        edges = torch.zeros(len(chains[-1]), batch, N - 1, C, C)
        for b in range(lengths.shape[0]):
            edges[: enum_lengths[b], b, : lengths[b] - 1] = model.to_parts(
                torch.stack([torch.tensor(c) for (c, _) in chains[lengths[b] - 1]]), C
            )

        return (
            semiring.unconvert(ret),
            [s for (_, s) in chains[-1]],
            edges,
            enum_lengths,
        )


class DepTreeTest:
    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

    @staticmethod
    def _rand():
        b = torch.randint(2, 4, (1,))
        N = torch.randint(2, 4, (1,))
        return torch.rand(b, N, N, 1), (b.item(), N.item())

    def enumerate(self, arc_scores, non_proj=False, multi_root=True):
        semiring = self.semiring
        parses = []
        q = []
        arc_scores = torch_struct.convert(arc_scores)
        batch, N, _, F = arc_scores.shape
        arc_scores = arc_scores.sum(-1)
        for mid in itertools.product(range(N + 1), repeat=N - 1):
            parse = [-1] + list(mid)
            if not _is_spanning(parse):
                continue
            if not non_proj and not _is_projective(parse):
                continue

            if not multi_root and _is_multi_root(parse):
                continue

            q.append(parse)
            parses.append(
                semiring.times(*[arc_scores[:, parse[i], i] for i in range(1, N, 1)])
            )
        return semiring.sum(torch.stack(parses, dim=-1)), None


class SemiMarkovTest:
    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

    # Tests

    @staticmethod
    def _rand():
        b = torch.randint(2, 4, (1,))
        N = torch.randint(2, 4, (1,))
        K = torch.randint(2, 4, (1,))
        C = torch.randint(2, 4, (1,))
        return torch.rand(b, N, K, C, C), (b.item(), (N + 1).item())

    def enumerate(self, edge):
        semiring = self.semiring
        ssize = semiring.size()
        batch, N, K, C, _ = edge.shape
        edge = semiring.convert(edge)
        chains = {}
        chains[0] = [
            ([(c, 0)], semiring.one_(torch.zeros(ssize, batch))) for c in range(C)
        ]

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
                                semiring.mul(
                                    score, edge[:, :, n - k, k, c, chain[-1][0]]
                                ),
                            )
                        )
        ls = [s for (_, s) in chains[N]]
        return semiring.unconvert(semiring.sum(torch.stack(ls, dim=1), dim=1)), ls


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


def _is_multi_root(parse):
    root_count = 0
    for m, h in enumerate(parse):
        if h == 0:
            root_count += 1
    return root_count > 1


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


class CKY_CRFTest:
    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

    # For testing
    def enumerate(self, scores):
        semiring = self.semiring
        batch, N, _, NT = scores.shape

        def enumerate(x, start, end):
            if start + 1 == end:
                yield (scores[:, start, start, x], [(start, x)])
            else:
                for w in range(start + 1, end):
                    for y in range(NT):
                        for z in range(NT):
                            for m1, y1 in enumerate(y, start, w):
                                for m2, z1 in enumerate(z, w, end):
                                    yield (
                                        semiring.times(
                                            m1, m2, scores[:, start, end - 1, x]
                                        ),
                                        [(x, start, w, end)] + y1 + z1,
                                    )

        ls = []
        for nt in range(NT):
            ls += [s for s, _ in enumerate(nt, 0, N)]

        return semiring.sum(torch.stack(ls, dim=-1)), None

    @staticmethod
    def _rand():
        batch = torch.randint(2, 5, (1,))
        N = torch.randint(2, 5, (1,))
        NT = torch.randint(2, 5, (1,))
        scores = torch.rand(batch, N, N, NT)
        return scores, (batch.item(), N.item())


class CKYTest:
    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

    def enumerate(self, scores):
        terms, rules, roots = scores
        semiring = self.semiring
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape

        def enumerate(x, start, end):
            if start + 1 == end:
                yield (terms[:, start, x - NT], [(start, x - NT)])
            else:
                for w in range(start + 1, end):
                    for y in range(NT) if w != start + 1 else range(NT, NT + T):
                        for z in range(NT) if w != end - 1 else range(NT, NT + T):
                            for m1, y1 in enumerate(y, start, w):
                                for m2, z1 in enumerate(z, w, end):
                                    yield (
                                        semiring.times(
                                            semiring.times(m1, m2), rules[:, x, y, z]
                                        ),
                                        [(x, start, w, end)] + y1 + z1,
                                    )

        ls = []
        for nt in range(NT):
            ls += [semiring.times(s, roots[:, nt]) for s, _ in enumerate(nt, 0, N)]
        return semiring.sum(torch.stack(ls, dim=-1)), None

    @staticmethod
    def _rand():
        batch = torch.randint(2, 5, (1,))
        N = torch.randint(2, 5, (1,))
        NT = torch.randint(2, 5, (1,))
        T = torch.randint(2, 5, (1,))
        terms = torch.rand(batch, N, T)
        rules = torch.rand(batch, NT, (NT + T), (NT + T))
        roots = torch.rand(batch, NT)
        return (terms, rules, roots), (batch.item(), N.item())


class AlignmentTest:
    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

    @staticmethod
    def _rand(min_n=2):
        b = torch.randint(2, 4, (1,))
        N = torch.randint(min_n, 4, (1,))
        M = torch.randint(min_n, 4, (1,))
        N = torch.min(M, N)
        return torch.rand(b, N, M, 3), (b.item(), (N).item())

    def enumerate(self, edge, lengths=None):
        semiring = self.semiring
        edge, batch, N, M, lengths = self._check_potentials(edge, lengths)
        d = {}
        d[0, 0] = [([(0, 0)], edge[:, :, 0, 0, 1])]
        # enum_lengths = torch.LongTensor(lengths.shape)
        for i in range(N):
            for j in range(M):
                d.setdefault((i + 1, j + 1), [])
                d.setdefault((i, j + 1), [])
                d.setdefault((i + 1, j), [])
                for chain, score in d[i, j]:
                    if i + 1 < N and j + 1 < M:
                        d[i + 1, j + 1].append(
                            (
                                chain + [(i + 1, j + 1)],
                                semiring.mul(score, edge[:, :, i + 1, j + 1, 1]),
                            )
                        )
                    if i + 1 < N:

                        d[i + 1, j].append(
                            (
                                chain + [(i + 1, j)],
                                semiring.mul(score, edge[:, :, i + 1, j, 2]),
                            )
                        )
                    if j + 1 < M:
                        d[i, j + 1].append(
                            (
                                chain + [(i, j + 1)],
                                semiring.mul(score, edge[:, :, i, j + 1, 0]),
                            )
                        )
        all_val = torch.stack([x[1] for x in d[N - 1, M - 1]], dim=-1)
        return semiring.unconvert(semiring.sum(all_val)), None


test_lookup = {
    torch_struct.LinearChain: LinearChainTest,
    torch_struct.SemiMarkov: SemiMarkovTest,
    torch_struct.DepTree: DepTreeTest,
    torch_struct.CKY_CRF: CKY_CRFTest,
    torch_struct.CKY: CKYTest,
    torch_struct.Alignment: AlignmentTest,
}
