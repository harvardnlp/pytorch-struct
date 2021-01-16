import torch
from .helpers import _Struct, Chart

A, B = 0, 1


class CKY(_Struct):
    def logpartition(self, scores, lengths=None, force_grad=False):

        semiring = self.semiring

        # Checks
        terms, rules, roots = scores
        rules.requires_grad_(True)
        ssize = semiring.size()
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape
        S = NT + T

        terms, rules, roots = (
            semiring.convert(terms).requires_grad_(True),
            semiring.convert(rules).requires_grad_(True),
            semiring.convert(roots).requires_grad_(True),
        )
        if lengths is None:
            lengths = torch.LongTensor([N] * batch).to(terms.device)

        # Charts
        beta = [Chart((batch, N, N, NT), rules, semiring) for _ in range(2)]
        span = [None for _ in range(N)]
        v = (ssize, batch)
        term_use = terms + 0.0

        # Split into NT/T groups
        NTs = slice(0, NT)
        Ts = slice(NT, S)
        rules = rules.view(ssize, batch, 1, NT, S, S)

        def arr(a, b):
            return rules[..., a, b].contiguous().view(*v + (NT, -1)).transpose(-2, -1)

        matmul = semiring.matmul
        times = semiring.times
        X_Y_Z = arr(NTs, NTs)
        X_Y1_Z = arr(Ts, NTs)
        X_Y_Z1 = arr(NTs, Ts)
        X_Y1_Z1 = arr(Ts, Ts)

        for w in range(1, N):
            all_span = []
            v2 = v + (N - w, -1)

            Y = beta[A][: N - w, :w, :]
            Z = beta[B][w:, N - w :, :]
            X1 = matmul(matmul(Y.transpose(-2, -1), Z).view(*v2), X_Y_Z)
            all_span.append(X1)

            Y_term = term_use[..., : N - w, :, None]
            Z_term = term_use[..., w:, None, :]

            Y = Y[..., -1, :].unsqueeze(-1)
            X2 = matmul(times(Y, Z_term).view(*v2), X_Y_Z1)

            Z = Z[..., 0, :].unsqueeze(-2)
            X3 = matmul(times(Y_term, Z).view(*v2), X_Y1_Z)
            all_span += [X2, X3]

            if w == 1:
                X4 = matmul(times(Y_term, Z_term).view(*v2), X_Y1_Z1)
                all_span.append(X4)

            span[w] = semiring.sum(torch.stack(all_span, dim=-1))
            beta[A][: N - w, w, :] = span[w]
            beta[B][w:N, N - w - 1, :] = span[w]

        final = beta[A][0, :, NTs]
        top = torch.stack([final[:, i, l - 1] for i, l in enumerate(lengths)], dim=1)
        log_Z = semiring.dot(top, roots)
        return log_Z, (term_use, rules, roots, span[1:])

    def marginals(self, scores, lengths=None, _autograd=True, _raw=False):
        """
        Compute the marginals of a CFG using CKY.

        Parameters:
            scores : terms : b x n x T
                     rules : b x NT x (NT+T) x (NT+T)
                     root:   b x NT
            lengths : lengths in batch

        Returns:
            v: b tensor of total sum
            spans: bxNxT terms, (bxNTx(NT+S)x(NT+S)) rules, bxNT roots

        """
        terms, rules, roots = scores
        batch, N, T = terms.shape
        _, NT, _, _ = rules.shape

        v, (term_use, rule_use, root_use, spans) = self.logpartition(
            scores, lengths=lengths, force_grad=True
        )

        def marginal(obj, inputs):
            obj = self.semiring.unconvert(obj).sum(dim=0)
            marg = torch.autograd.grad(
                obj,
                inputs,
                create_graph=True,
                only_inputs=True,
                allow_unused=False,
            )

            spans_marg = torch.zeros(
                batch, N, N, NT, dtype=scores[1].dtype, device=scores[1].device
            )
            span_ls = marg[3:]
            for w in range(len(span_ls)):
                x = span_ls[w].sum(dim=0, keepdim=True)
                spans_marg[:, w, : N - w - 1] = self.semiring.unconvert(x)

            rule_marg = self.semiring.unconvert(marg[0]).squeeze(1)
            root_marg = self.semiring.unconvert(marg[1])
            term_marg = self.semiring.unconvert(marg[2])

            assert term_marg.shape == (batch, N, T)
            assert root_marg.shape == (batch, NT)
            assert rule_marg.shape == (batch, NT, NT + T, NT + T)
            return (term_marg, rule_marg, root_marg, spans_marg)

        inputs = (rule_use, root_use, term_use) + tuple(spans)
        if _raw:
            paths = []
            for k in range(v.shape[0]):
                obj = v[k : k + 1]
                marg = marginal(obj, inputs)
                paths.append(marg[-1])
            paths = torch.stack(paths, 0)
            obj = v.sum(dim=0, keepdim=True)
            term_marg, rule_marg, root_marg, _ = marginal(obj, inputs)
            return term_marg, rule_marg, root_marg, paths
        else:
            return marginal(v, inputs)

    def score(self, potentials, parts):
        terms, rules, roots = potentials[:3]
        m_term, m_rule, m_root = parts[:3]
        b = m_term.shape[0]
        return (
            m_term.mul(terms).view(b, -1).sum(-1)
            + m_rule.mul(rules).view(b, -1).sum(-1)
            + m_root.mul(roots).view(b, -1).sum(-1)
        )

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
        assert terms.shape[1] == N

        spans = torch.zeros(batch, N, N, S, dtype=rules.dtype, device=rules.device)
        rules = rules.sum(dim=-1).sum(dim=-1)
        for n in range(N):
            spans[:, torch.arange(N - n - 1), torch.arange(n + 1, N), :NT] = rules[
                :, n, torch.arange(N - n - 1)
            ]
        spans[:, torch.arange(N), torch.arange(N), NT:] = terms
        return spans, (NT, S - NT)

    @staticmethod
    def _intermediary(spans):
        batch, N = spans.shape[:2]
        splits = {}
        cover = spans.nonzero()
        left, right = {}, {}
        for k in range(cover.shape[0]):
            b, i, j, A = cover[k].tolist()
            left.setdefault((b, i), [])
            right.setdefault((b, j), [])
            left[b, i].append((A, j, j - i + 1))
            right[b, j].append((A, i, j - i + 1))

        for x in range(cover.shape[0]):
            b, i, j, A = cover[x].tolist()
            if i == j:
                continue
            b_final = None
            c_final = None
            k_final = None
            for B_p, k, a_span in left.get((b, i), []):
                if k > j:
                    continue
                for C_p, k_2, b_span in right.get((b, j), []):
                    if k_2 == k + 1 and a_span + b_span == j - i + 1:
                        k_final = k
                        b_final = B_p
                        c_final = C_p
                        break
                if b_final is not None:
                    break
            assert k_final is not None, "%s %s %s %s" % (b, i, j, spans[b].nonzero())
            splits[(b, i, j)] = k_final, b_final, c_final
        return splits

    @classmethod
    def to_networkx(cls, spans):
        cur = 0
        N = spans.shape[1]
        n_nodes = int(spans.sum().item())
        cover = spans.nonzero().cpu()
        order = torch.argsort(cover[:, 2] - cover[:, 1])
        left = {}
        right = {}
        ordered = cover[order]
        label = ordered[:, 3]
        a = []
        b = []
        topo = [[] for _ in range(N)]
        for n in ordered:
            batch, i, j, _ = n.tolist()
            # G.add_node(cur, label=A)
            if i - j != 0:
                a.append(left[(batch, i)][0])
                a.append(right[(batch, j)][0])
                b.append(cur)
                b.append(cur)
                order = max(left[(batch, i)][1], right[(batch, j)][1]) + 1
            else:
                order = 0
            left[(batch, i)] = (cur, order)
            right[(batch, j)] = (cur, order)
            topo[order].append(cur)
            cur += 1
        indices = left
        return (n_nodes, a, b, label), indices, topo
