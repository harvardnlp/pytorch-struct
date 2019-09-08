import torch
from .helpers import _Struct
from .semirings import LogSemiring
from torch.autograd import Function

A, B = 0, 1


class DPManual2(Function):
    @staticmethod
    def forward(ctx, obj, terms, rules, roots, lengths):
        with torch.no_grad():
            v, _, alpha = obj._dp((terms, rules, roots), lengths, False)
        ctx.obj = obj
        ctx.lengths = lengths
        ctx.alpha = alpha
        ctx.v = v
        ctx.save_for_backward(terms, rules, roots)
        return v

    @staticmethod
    def backward(ctx, grad_v):
        terms, rules, roots = ctx.saved_tensors
        with torch.no_grad():
            marginals = ctx.obj._dp_backward(
                (terms, rules, roots), ctx.lengths, ctx.alpha, ctx.v
            )
        return None, marginals[0], marginals[1].sum(1).sum(1), marginals[2], None


class CKY(_Struct):
    def sum(self, scores, lengths=None, force_grad=False, _autograd=True):
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
        if _autograd or self.semiring is not LogSemiring:
            return self._dp(scores, lengths)[0]
        else:
            return DPManual2.apply(self, *scores, lengths)

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
        term_use = self._make_chart(1, (batch, N, T), terms, force_grad)[0]
        term_use[:] = terms + 0.0
        beta[A][:, :, 0, NT:] = term_use
        beta[B][:, :, N - 1, NT:] = term_use
        for w in range(1, N):
            Y = beta[A][:, : N - w, :w, :].view(batch, N - w, w, 1, S, 1)
            Z = beta[B][:, w:, N - w :, :].view(batch, N - w, w, 1, 1, S)
            Y, Z = Y.clone(), Z.clone()
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
        return log_Z, (term_use, rule_use, top), beta

    def _dp_backward(self, scores, lengths, alpha_in, v, force_grad=False):
        terms, rules, roots = scores
        semiring = self.semiring
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape
        S = NT + T
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)

        beta = self._make_chart(2, (batch, N, N, NT + T), rules, force_grad)
        span_l = self._make_chart(N, (batch, N, NT + T), rules, force_grad)
        span_r = self._make_chart(N, (batch, N, NT + T), rules, force_grad)
        term_use = self._make_chart(1, (batch, N, T), terms, force_grad)[0]

        ssum = semiring.sum
        st = semiring.times
        X_Y_Z = rules.view(batch, 1, NT, S, S)

        for w in range(N - 1, -1, -1):
            for b, l in enumerate(lengths):
                beta[A][b, 0, l - 1, :NT] = roots[b]
                beta[B][b, l - 1, N - (l), :NT] = roots[b]

            # LEFT
            # all bigger on the left.
            X = beta[A][:, : N - w - 1, w + 1 :, :NT].view(
                batch, N - w - 1, N - w - 1, NT, 1, 1
            )
            Z = alpha_in[A][:, w + 1 : N, 0 : N - w - 1].view(
                batch, N - w - 1, N - w - 1, 1, 1, S
            )
            t = st(ssum(st(X, Z), dim=2), X_Y_Z)
            # sum out x and y
            span_l[w] = ssum(ssum(t, dim=-3), dim=-1)

            # RIGHT
            X = beta[B][:, w + 1 :, : N - 1 - w, :NT].view(
                batch, N - w - 1, N - w - 1, NT, 1, 1
            )
            Y = alpha_in[B][:, : N - w - 1, w + 1 :, :].view(
                batch, N - w - 1, N - w - 1, 1, S, 1
            )
            t = st(ssum(st(X, Y), dim=2), X_Y_Z)

            span_r[w] = ssum(ssum(t, dim=-3), dim=-2)

            beta[A][:, : N - w - 1, w, :] = span_l[w]
            beta[A][:, 1 : N - w, w, :] = ssum(
                torch.stack([span_r[w], beta[A][:, 1 : N - w, w, :]]), dim=0
            )
            beta[B][:, w:, N - w - 1, :] = beta[A][:, : N - w, w, :]

        term_use[:, :, :] = st(beta[A][:, :, 0, NT:], terms)
        term_marginals = self._make_chart(1, (batch, N, T), terms, force_grad=False)[0]
        for n in range(N):
            term_marginals[:, n] = semiring.div_exp(term_use[:, n], v.view(batch, 1))

        root_marginals = self._make_chart(1, (batch, NT), terms, force_grad=False)[0]
        for b in range(batch):
            root_marginals[b] = semiring.div_exp(
                st(alpha_in[A][b, 0, lengths[b] - 1, :NT], roots[b]), v[b].view(1)
            )
        edge_marginals = self._make_chart(
            1, (batch, N, N, NT, S, S), terms, force_grad=False
        )[0]
        edge_marginals.fill_(0)
        for w in range(1, N):
            Y = alpha_in[A][:, : N - w, :w, :].view(batch, N - w, w, 1, S, 1)
            Z = alpha_in[B][:, w:, N - w :, :].view(batch, N - w, w, 1, 1, S)
            score = semiring.times(semiring.sum(semiring.times(Y, Z), dim=2), X_Y_Z)
            score = st(score, beta[A][:, : N - w, w, :NT].view(batch, N - w, NT, 1, 1))
            edge_marginals[:, : N - w, w - 1] = semiring.div_exp(
                score, v.view(batch, 1, 1, 1, 1)
            )
        edge_marginals = edge_marginals.transpose(1, 2)

        return (term_marginals, edge_marginals, root_marginals)

    def marginals(self, scores, lengths=None, _autograd=True):
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
        v, (term_use, rule_use, top), alpha = self._dp(
            scores, lengths=lengths, force_grad=True
        )
        if _autograd or self.semiring is not LogSemiring:
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
        else:
            return self._dp_backward(scores, lengths, alpha, v)

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

        ls = []
        for nt in range(NT):
            ls += [semiring.times(s, roots[:, nt]) for s, _ in enumerate(nt, 0, N)]
        return semiring.sum(torch.stack(ls, dim=-1))

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

    def score(self, potentials, parts):
        terms, rules, roots = potentials
        m_term, m_rule, m_root = parts
        b = m_term.shape[0]
        return (
            m_term.mul(terms).view(b, -1).sum(-1)
            + m_rule.sum(dim=1).sum(dim=1).mul(rules).view(b, -1).sum(-1)
            + m_root.mul(roots).view(b, -1).sum(-1)
        )
