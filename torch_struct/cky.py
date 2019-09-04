import torch
from .helpers import _Struct

A, B = 0, 1


class CKY(_Struct):
    def sum(self, scores, lengths=None, force_grad=False):
        """
        Compute the inside pass of a CFG using CKY.

        Parameters:
            terms : b x n x T
            rules : b x NT x (NT+T) x (NT+T)
            root:   b x NT

        Returns:
            v: b tensor of total sum
            spans: list of N,  b x N x (NT+t)
        """
        return self._dp(scores, lengths)[0]

    def _dp(self, scores, lengths=None, force_grad=False):
        terms, rules, roots = scores
        semiring = self.semiring
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape
        S = NT + T

        if lengths is None:
            lengths = torch.LongTensor([N] * batch)
        beta = self._make_chart(2, (batch, N, N, NT + T), rules, force_grad)
        span = self._make_chart(N, (batch, N, NT + T), rules, force_grad)
        rule_use = [
            self._make_chart(1, (batch, N - w - 1, NT, S, S), rules, force_grad)[0]
            for w in range(N - 1)
        ]
        top = self._make_chart(1, (batch, NT), rules, force_grad)[0]
        term_use = self._make_chart(1, (batch, N, T), rules, force_grad)[
            0
        ].requires_grad_(True)
        term_use[:] = terms
        beta[A][:, :, 0, NT:] = term_use
        beta[B][:, :, N - 1, NT:] = term_use

        for w in range(1, N):
            Y = beta[A][:, : N - w, :w, :].view(batch, N - w, w, 1, S, 1)
            Z = beta[B][:, w:, N - w :, :].view(batch, N - w, w, 1, 1, S)
            X_Y_Z = rules.view(batch, 1, NT, S, S)
            rule_use[w - 1][:] = semiring.times(
                semiring.sum(semiring.times(Y, Z), dim=2), X_Y_Z
            )
            rulesmid = rule_use[w - 1].view(batch, N - w, NT, S * S)
            span[w] = semiring.sum(rulesmid, dim=3)
            beta[A][:, : N - w, w, :NT] = span[w]
            beta[B][:, w:N, N - w - 1, :NT] = beta[A][:, : N - w, w, :NT]

        top[:] = torch.stack([beta[A][i, 0, l - 1, :NT] for i, l in enumerate(lengths)])
        log_Z = semiring.dot(top, roots)
        return log_Z, (term_use, rule_use, top)

    def marginals(self, scores, lengths=None):
        """
        Compute the marginals of a CFG using CKY.

        Parameters:
            terms : b x n x T
            rules : b x NT x (NT+T) x (NT+T)
            root:   b x NT

        Returns:
            v: b tensor of total sum
            spans: bxNxT terms, (bxNxNxNTxSxS) rules, bxNT roots

        """
        terms, rules, roots = scores
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape
        S = NT + T
        v, (term_use, rule_use, top) = self._dp(scores, lengths=lengths)
        marg = torch.autograd.grad(
            v.sum(dim=0),
            tuple(rule_use) + (top, term_use),
            create_graph=True,
            only_inputs=True,
            allow_unused=False,
        )
        rule_use = marg[:-2]
        rules = torch.zeros(batch, N, N, NT, S, S)
        for w in range(len(rule_use)):
            rules[:, w, : N - w - 1] = rule_use[w]
        assert marg[-1].shape == (batch, N, T)
        assert marg[-2].shape == (batch, NT)
        return (marg[-1], rules, marg[-2])

    @staticmethod
    def to_parts(spans, extra, lengths=None):
        NT, T = extra

        batch, N, N, S = spans.shape
        assert S == NT + T
        terms = torch.zeros(batch, N, T)
        rules = torch.zeros(batch, NT, S, S)
        roots = torch.zeros(batch, NT)
        for b in range(batch):
            roots[b, :] = spans[b, 0, lengths[b] - 1, :NT]
            terms[b, : lengths[b]] = spans[
                b, torch.arange(lengths[b]), torch.arange(lengths[b]), NT:
            ]
            cover = spans[b].nonzero()

            left = {i: [] for i in range(N)}
            right = {i: [] for i in range(N)}
            for i in range(cover.shape[0]):
                i, j, A = cover[i].tolist()
                left[i].append((A, j, j - i + 1))
                right[j].append((A, i, j - i + 1))
            for i in range(cover.shape[0]):
                i, j, A = cover[i].tolist()
                B = None
                for B_p, k, a_span in left[i]:
                    for C_p, k_2, b_span in right[j]:
                        if k_2 == k + 1 and a_span + b_span == j - i + 1:
                            B, C = B_p, C_p
                            break
                if j > i:
                    assert B is not None, "%s" % ((i, j, left[i], right[j], cover),)
                    rules[b, A, B, C] += 1
        return terms, rules, roots

    @staticmethod
    def from_parts(chart):
        terms, rules, roots = chart
        batch, N, N, NT, S, S = rules.shape
        spans = torch.zeros(batch, N, N, S)
        rules = rules.sum(dim=-1).sum(dim=-1)

        for n in range(N):
            spans[:, torch.arange(N - n - 1), torch.arange(n + 1, N), :NT] = rules[
                :, n, torch.arange(N - n - 1)
            ]
        spans[:, torch.arange(N), torch.arange(N), NT:] = terms
        print(rules.nonzero(), spans.nonzero())
        return spans, (NT, S - NT)

    ###### Test

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

        # for nt in range(NT):
        #     print(list(enumerate(nt, 0, N)))
        ls = []
        for nt in range(NT):
            ls += [semiring.times(s, roots[:, nt]) for s, _ in enumerate(nt, 0, N)]
        return semiring.sum(torch.stack(ls, dim=-1))

    @staticmethod
    def _rand():
        batch = torch.randint(2, 4, (1,))
        N = torch.randint(2, 4, (1,))
        NT = torch.randint(2, 4, (1,))
        T = torch.randint(2, 4, (1,))
        terms = torch.rand(batch, N, T)
        rules = torch.rand(batch, NT, (NT + T), (NT + T))
        roots = torch.rand(batch, NT)
        return (terms, rules, roots), (batch.item(), N.item())

    def score(self, potentials, parts):
        terms, rules, roots = potentials
        m_term, m_rule, m_root = parts
        b = m_term.shape[0]
        return (
            m_term.mul(terms).view(b, -1).sum(-1)
            + m_rule.sum(dim=1).sum(dim=1).mul(rules).view(b, -1).sum(-1)
            + m_root.mul(roots).view(b, -1).sum(-1)
        )
