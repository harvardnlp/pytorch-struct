import torch
from .helpers import _Struct, Chart

A, B = 0, 1


class Full_CKY_CRF(_Struct):
    def _check_potentials(self, edge, lengths=None):
        batch, N, N1, N2, NT, NT1, NT2 = self._get_dimension(edge)
        assert (
            N == N1 == N2 and NT == NT1 == NT2
        ), f"Want N:{N} == N1:{N1} == N2:{N2} and NT:{NT} == NT1:{NT1} == NT2:{NT2}"
        edge = self.semiring.convert(edge)
        semiring_shape = edge.shape[:-7]
        if lengths is None:
            lengths = torch.LongTensor([N] * batch).to(edge.device)

        return edge, semiring_shape, batch, N, NT, lengths

    def _dp(self, scores, lengths=None, force_grad=False, cache=True):
        semiring = self.semiring
        scores, sshape, batch, N, NT, lengths = self._check_potentials(scores, lengths)
        # scores.shape = *sshape, B, N, N, N, NT, NT, NT

        beta = [Chart((batch, N, N), scores, semiring, cache=cache) for _ in range(2)]
        L_DIM, R_DIM = len(sshape) + 1, len(sshape) + 2  # usually 2,3

        # Initialize
        reduced_scores = semiring.sum(scores)
        print(reduced_scores.shape)
        term = reduced_scores.diagonal(0, L_DIM, R_DIM)
        bs, ns, nts = torch.arange(batch), torch.arange(N), torch.arange(NT)
        term_scores = scores[:, bs, ns, ns, ns, nts, nts, nts]
        print(term_scores.shape)
        beta[A][ns, 0] = term
        beta[B][ns, N - 1] = term

        # Run
        for w in range(1, N):
            left = slice(None, N - w)
            right = slice(w, None)
            Y = beta[A][left, :w]
            Z = beta[B][right, N - w :]
            score = reduced_scores.diagonal(w, L_DIM, R_DIM)
            new = semiring.times(semiring.dot(Y, Z), score)
            beta[A][left, w] = new
            beta[B][right, N - w - 1] = new

        final = beta[A][0, :]
        log_Z = final[:, torch.arange(batch), lengths - 1]
        return log_Z, [scores], beta

    # For testing

    def enumerate(self, scores, lengths=None):
        semiring = self.semiring
        batch, N, _, _, NT, _, _ = scores.shape

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
                                        semiring.times(m1, m2, scores[:, start, end - 1, x]),
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
        scores = torch.rand(batch, N, N, N, NT, NT, NT)
        return scores, (batch.item(), N.item())
