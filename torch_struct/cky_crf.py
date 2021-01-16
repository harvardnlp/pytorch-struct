import torch
from .helpers import _Struct, Chart

A, B = 0, 1


class CKY_CRF(_Struct):
    def _check_potentials(self, edge, lengths=None):
        batch, N, _, NT = self._get_dimension(edge)
        edge = self.semiring.convert(edge)
        if lengths is None:
            lengths = torch.LongTensor([N] * batch).to(edge.device)

        return edge, batch, N, NT, lengths

    def logpartition(self, scores, lengths=None, force_grad=False):
        semiring = self.semiring
        scores, batch, N, NT, lengths = self._check_potentials(scores, lengths)

        beta = [Chart((batch, N, N), scores, semiring) for _ in range(2)]
        L_DIM, R_DIM = 2, 3

        # Initialize
        reduced_scores = semiring.sum(scores)
        term = reduced_scores.diagonal(0, L_DIM, R_DIM)
        ns = torch.arange(N)
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
        return log_Z, [scores]
