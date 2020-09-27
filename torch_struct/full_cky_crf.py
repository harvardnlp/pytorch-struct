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
        sr = self.semiring
        # torch.autograd.set_detect_anomaly(True)

        # Scores.shape = *sshape, B, N, N, N, NT, NT, NT
        # w/ semantics [ *semiring stuff, b, i, j, k, A, B, C]
        # where b is batch index, i is left endpoint, j is right endpoint, k is splitpoint,  with rule A -> B C
        scores, sshape, batch, N, NT, lengths = self._check_potentials(scores, lengths)
        sshape, sdims = list(sshape), list(range(len(sshape)))  # usually [0]
        S, b = len(sdims), batch

        # Initialize data structs
        LEFT, RIGHT = 0, 1
        L_DIM, R_DIM = S + 1, S + 2  # one and two to the right of the batch dim
        # Will store sum of subtrees up to i,j,A from the left and right
        # beta[LEFT][i,d,A] = sum of potentials of all subtrees in span i,j=(i+d) with nonterminal A
        #                     indexed from the left endpoint i plus the width d
        # .                 = alpha[i,j=(i+d),A] in a nonvectorized version
        # beta[RIGHT][j,d',A] = sum of potentials of all subtrees in span i=(j-(N-d')),j with NT A
        #                       indexed from the right endpoint, from widest to shortest subtrees.
        #                       This gets filled in from right to left.

        # OVERRIDE CACHE
        cache = False
        # print("cache", cache)
        beta = [Chart((b, N, N, NT), scores, sr, cache=cache) for _ in range(2)]

        # Initialize the base cases with scores from diagonal i=j=k, A=B=C
        term_scores = (
            scores.diagonal(0, L_DIM, R_DIM)  # diag i,j now at dim -1
            .diagonal(0, L_DIM, -1)  # diag of k with that gives i=j=k, now at dim -1
            .diagonal(0, -4, -3)  # diag of A, B, now at dim -1, ijk moves to -2
            .diagonal(0, -3, -1)  # diag of C with that gives A=B=C
        )
        assert term_scores.shape[S + 1 :] == (N, NT), f"{term_scores.shape[S + 1 :]} == {(N, NT)}"
        beta[LEFT][:, 0, :] = term_scores
        beta[RIGHT][:, N - 1, :] = term_scores
        alpha_left = term_scores
        alpha_right = term_scores

        ### old: init with semiring's multiplicative identity, gives zeros mass to leaves
        # ns = torch.arange(NT)
        # beta[LEFT][:, 0, :] = sr.one_(beta[LEFT][:, 0, :])
        # beta[RIGHT][:, N - 1, :] = sr.one_(beta[RIGHT][:, N - 1, :])
        # alpha_left = sr.one_(torch.ones(sshape + [b, N, NT]).to(scores.device))
        # alpha_right = sr.one_(torch.ones(sshape + [b, N, NT]).to(scores.device))

        alphas = [[alpha_left], [alpha_right]]

        # Run vectorized inside alg
        for w in range(1, N):
            # print("\nw", w, "N-w", N - w)
            # Scores
            # What we want is a tensor with:
            #  shape: *sshape, batch, (N-w), NT, w, NT, NT
            #  w/ semantics: [...batch, (i,j=i+w), A, k, B, C]
            #  where (i,j=i+w) means the diagonal of trees nodes with width w
            # Shape: *sshape, batch, N, NT, NT, NT, (N-w) w/ semantics [ ...batch, k, A, B, C, (i,j=i+w)]
            score = scores.diagonal(w, L_DIM, R_DIM)  # get diagonal scores
            # print("diagonal", score.shape[S:])

            score = score.permute(sdims + [-6, -1, -4, -5, -3, -2])  # move diag (-1) dim and head NT (-4) dim to front
            # print("permute", score.shape[S:])
            score = score[..., :w, :, :]  # remove illegal splitpoints
            # print("slice", score.shape[S:])
            assert score.shape[S:] == (batch, N - w, NT, w, NT, NT), f"{score.shape[S:]} == {(b, N-w, NT, w, NT, NT)}"
            # print("S", score[0, 0, :, 0, :, 0, 0].exp())
            # Sums of left subtrees
            # Shape: *sshape, batch, (N-w), w, NT
            # where L[..., i, d, B] is the sum of subtrees up to (i,j=(i+d),B)
            left = slice(None, N - w)  # left indices
            L1 = beta[LEFT][left, :w]
            L = torch.stack(alphas[LEFT][:w], dim=-2)[..., left, :, :]

            assert L.isclose(L1).all()
            # print("L", L.shape)

            # Sums of right subtrees
            # Shape: *sshape, batch, (N-w), w, NT
            # where R[..., h, d, C] is the sum of subtrees up to (i=(N-h-d),j=(N-h),C)
            right = slice(w, None)  # right indices
            R1 = beta[RIGHT][right, N - w :]
            R = torch.stack(list(reversed(alphas[RIGHT][:w])), dim=-2)[..., right, :, :]
            assert R.isclose(R1).all()
            # print("R", R.shape)  # R[0, 0, :, :, 0].exp())

            # Broadcast them both to match missing dims in score
            # Left B is duplicated for all head and right symbols A C
            L_bcast = L.reshape(list(sshape) + [b, N - w, 1, w, NT, 1]).repeat(S * [1] + [1, 1, NT, 1, 1, NT])
            # Right C is duplicated for all head and left symbols A B
            R_bcast = R.reshape(list(sshape) + [b, N - w, 1, w, 1, NT]).repeat(S * [1] + [1, 1, NT, 1, NT, 1])

            assert score.shape == L_bcast.shape == R_bcast.shape == tuple(list(sshape) + [b, N - w, NT, w, NT, NT])
            # print(score.shape[S + 1 :], L_bcast.shape, R_bcast.shape)

            # Now multiply all the scores and sum over k, B, C dimensions (the last three dims)
            assert sr.times(score, L_bcast, R_bcast).shape == tuple(list(sshape) + [b, N - w, NT, w, NT, NT])
            sum_prod_w = sr.sum(sr.sum(sr.sum(sr.times(score, L_bcast, R_bcast))))
            # print("sum prod w", sum_prod_w.exp())
            assert sum_prod_w.shape[S:] == (b, N - w, NT), f"{sum_prod_w.shape[S:]} == {(b,N-w, NT)}"

            #     new = sr.times(sr.dot(Y, Z), score)
            beta[LEFT][left, w] = sum_prod_w
            beta[RIGHT][right, N - w - 1] = sum_prod_w
            # pad = sr.zero_(torch.ones_like(sum_prod_w))[..., :w, :]
            pad = sr.zero_(torch.ones(sshape + [b, w, NT]).to(sum_prod_w.device))
            sum_prod_w_left = torch.cat([sum_prod_w, pad], dim=-2)
            sum_prod_w_right = torch.cat([pad, sum_prod_w], dim=-2)
            # print(sum_prod_w.shape, sum_prod_w_left.shape, sum_prod_w_right.shape)
            alphas[LEFT].append(sum_prod_w_left)
            alphas[RIGHT].append(sum_prod_w_right)
        # for c in range(NT):
        #     print(f"left c:{c}\n", beta[LEFT][:, :].exp().detach().numpy())

        #     print(f"right c:{c}\n", beta[RIGHT][:, :].exp().detach().numpy())

        final1 = sr.sum(beta[LEFT][0, :, :])
        final = sr.sum(torch.stack(alphas[LEFT], dim=-2))[..., 0, :]  # sum out root symbol
        # print(f"f1:{final1.shape}, f:{final.shape}, ls:{lengths}")
        assert final.isclose(final1).all(), f"final:\n{final}\nfinal1:\n{final1}"

        # log_Z = final[..., 0, lengths - 1]
        log_Z = final[:, torch.arange(batch), lengths - 1]
        # log_Z.exp().sum().backward()
        # print("Z", log_Z.exp())
        return log_Z, [scores], beta

    # For testing

    def enumerate(self, scores, lengths=None):
        raise NotImplementedError
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
