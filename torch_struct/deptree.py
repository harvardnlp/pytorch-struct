import torch
from .helpers import _Struct, Chart


def convert(logits):
    "move root arcs from diagonal"
    new_shape = list(logits.shape)
    new_shape[1] += 1
    new_shape[2] += 1
    new_logits = torch.zeros(new_shape).type_as(logits.data)
    new_logits.fill_(-1e9)
    new_logits[:, 1:, 1:] = logits

    N = logits.size(1)
    new_logits[:, 0, 1:] = logits[:, torch.arange(N), torch.arange(N)]
    new_logits[:, torch.arange(1, N + 1), torch.arange(1, N + 1)] = -1e9
    return new_logits


def unconvert(logits):
    "Move root arcs to diagonal"
    new_shape = list(logits.shape)
    new_shape[1] -= 1
    new_shape[2] -= 1
    new_logits = torch.zeros(new_shape, dtype=logits.dtype, device=logits.device)

    new_logits.fill_(-1e9)
    new_logits[:, :, :] = logits[:, 1:, 1:]
    N = new_logits.size(1)
    new_logits[:, torch.arange(N), torch.arange(N)] = logits[:, 0, 1:]
    return new_logits


# Constants
A, B, R, C, L, I = 0, 1, 1, 1, 0, 0


class DepTree(_Struct):
    """
    A projective dependency CRF.

    Parameters:
        arc_scores_in: Arc scores of shape (B, N, N) or (B, N, N, L) with root scores on
        diagonal.

    Note: For single-root case, do not set cache=True for now.
    """

    def logpartition(self, arc_scores_in, lengths=None, force_grad=False):
        multiroot = getattr(self, "multiroot", True)
        if arc_scores_in.dim() not in (3, 4):
            raise ValueError("potentials must have dim of 3 (unlabeled) or 4 (labeled)")

        labeled = arc_scores_in.dim() == 4
        semiring = self.semiring
        arc_scores_in = convert(arc_scores_in)
        arc_scores_in, batch, N, lengths = self._check_potentials(
            arc_scores_in, lengths
        )
        arc_scores_in.requires_grad_(True)
        arc_scores = semiring.sum(arc_scores_in) if labeled else arc_scores_in
        alpha = [
            [
                [Chart((batch, N, N), arc_scores, semiring) for _ in range(2)]
                for _ in range(2)
            ]
            for _ in range(2)
        ]
        mask = torch.zeros(alpha[A][C][L].data.shape).bool()
        mask[:, :, :, 0].fill_(True)
        alpha[A][C][L].data[:] = semiring.fill(
            alpha[A][C][L].data[:], mask, semiring.one
        )
        alpha[A][C][R].data[:] = semiring.fill(
            alpha[A][C][R].data[:], mask, semiring.one
        )
        mask = torch.zeros(alpha[B][C][L].data[:].shape).bool()
        mask[:, :, :, -1].fill_(True)
        alpha[B][C][L].data[:] = semiring.fill(
            alpha[B][C][L].data[:], mask, semiring.one
        )
        alpha[B][C][R].data[:] = semiring.fill(
            alpha[B][C][R].data[:], mask, semiring.one
        )

        if multiroot:
            start_idx = 0
        else:
            start_idx = 1

        for k in range(1, N - start_idx):
            f = torch.arange(start_idx, N - k), torch.arange(k + start_idx, N)
            ACL = alpha[A][C][L][start_idx : N - k, :k]
            ACR = alpha[A][C][R][start_idx : N - k, :k]
            BCL = alpha[B][C][L][k + start_idx :, N - k :]
            BCR = alpha[B][C][R][k + start_idx :, N - k :]
            x = semiring.dot(ACR, BCL)
            arcs_l = semiring.times(x, arc_scores[:, :, f[1], f[0]])
            alpha[A][I][L][start_idx : N - k, k] = arcs_l
            alpha[B][I][L][k + start_idx : N, N - k - 1] = arcs_l
            arcs_r = semiring.times(x, arc_scores[:, :, f[0], f[1]])
            alpha[A][I][R][start_idx : N - k, k] = arcs_r
            alpha[B][I][R][k + start_idx : N, N - k - 1] = arcs_r
            AIR = alpha[A][I][R][start_idx : N - k, 1 : k + 1]
            BIL = alpha[B][I][L][k + start_idx :, N - k - 1 : N - 1]
            new = semiring.dot(ACL, BIL)
            alpha[A][C][L][start_idx : N - k, k] = new
            alpha[B][C][L][k + start_idx : N, N - k - 1] = new
            new = semiring.dot(AIR, BCR)
            alpha[A][C][R][start_idx : N - k, k] = new
            alpha[B][C][R][k + start_idx : N, N - k - 1] = new

        if not multiroot:
            root_incomplete_span = semiring.times(
                alpha[A][C][L][1, : N - 1], arc_scores[:, :, 0, 1:]
            )
            for k in range(1, N):
                AIR = root_incomplete_span[:, :, :k]
                BCR = alpha[B][C][R][k, N - k :]
                alpha[A][C][R][0, k] = semiring.dot(AIR, BCR)

        final = alpha[A][C][R][(0,)]
        v = torch.stack([final[:, i, l] for i, l in enumerate(lengths)], dim=1)
        return v, [arc_scores_in]

    def _check_potentials(self, arc_scores, lengths=None):
        semiring = self.semiring
        batch, N, N2, *_ = self._get_dimension(arc_scores)
        assert N == N2, "Non-square potentials"
        if lengths is None:
            lengths = torch.LongTensor([N - 1] * batch).to(arc_scores.device)
        assert max(lengths) <= N, "Length longer than N"
        arc_scores = semiring.convert(arc_scores)

        # Set the extra elements of the log-potentials to zero.
        keep = torch.ones_like(arc_scores).bool()
        for b in range(batch):
            keep[:, b, lengths[b] + 1 :, :].fill_(0.0)
            keep[:, b, :, lengths[b] + 1 :].fill_(0.0)
        arc_scores = semiring.fill(arc_scores, ~keep, semiring.zero)
        return arc_scores, batch, N, lengths

    def _arrange_marginals(self, grads):
        return self.semiring.convert(unconvert(self.semiring.unconvert(grads[0])))

    @staticmethod
    def to_parts(sequence, extra=None, lengths=None):
        """
        Convert a sequence representation to arcs

        Parameters:
            sequence : b x N long tensor in [0, N] (indexing is +1)
            extra : None
            lengths : lengths of sequences

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
        return unconvert(labels)

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


def deptree_part(arc_scores, multi_root, lengths=None, eps=1e-5):
    if lengths is not None:
        batch, N, N = arc_scores.shape
        x = torch.arange(N, device=arc_scores.device).expand(batch, N)
        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=arc_scores.device)
        lengths = lengths.unsqueeze(1)
        x = x < lengths
        det_offset = torch.diag_embed((~x).float())
        x = x.unsqueeze(2).expand(-1, -1, N)
        mask = torch.transpose(x, 1, 2) * x
        mask = mask.float()
        mask[mask == 0] = float("-inf")
        mask[mask == 1] = 0
        arc_scores = arc_scores + mask
    input = arc_scores
    eye = torch.eye(input.shape[1], device=input.device)
    laplacian = input.exp() + eps
    lap = laplacian.masked_fill(eye != 0, 0)
    lap = -lap + torch.diag_embed(lap.sum(1), offset=0, dim1=-2, dim2=-1)
    if lengths is not None:
        lap += det_offset

    if multi_root:
        rss = torch.diagonal(input, 0, -2, -1).exp()  # root selection scores
        lap = lap + torch.diag_embed(rss, offset=0, dim1=-2, dim2=-1)
    else:
        lap[:, 0] = torch.diagonal(input, 0, -2, -1).exp()
    return lap.logdet()


def deptree_nonproj(arc_scores, multi_root, lengths=None, eps=1e-5):
    """
    Compute the marginals of a non-projective dependency tree using the
    matrix-tree theorem.

    Allows for overlapping arcs.

    Much faster, but cannot provide a semiring.

    Parameters:
         arc_scores : b x N x N arc scores with root scores on diagonal.
         multi_root (bool) : multiple roots
         lengths : length of examples
         eps (float) : given

    Returns:
         arc_marginals : b x N x N.
    """
    if lengths is not None:
        batch, N, N = arc_scores.shape
        x = torch.arange(N, device=arc_scores.device).expand(batch, N)
        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=arc_scores.device)
        lengths = lengths.unsqueeze(1)
        x = x < lengths
        det_offset = torch.diag_embed((~x).float())
        x = x.unsqueeze(2).expand(-1, -1, N)
        mask = torch.transpose(x, 1, 2) * x
        mask = mask.float()
        mask[mask == 0] = float("-inf")
        mask[mask == 1] = 0
        arc_scores = arc_scores + mask

    input = arc_scores
    eye = torch.eye(input.shape[1], device=input.device)
    laplacian = input.exp() + eps
    lap = laplacian.masked_fill(eye != 0, 0)
    lap = -lap + torch.diag_embed(lap.sum(1), offset=0, dim1=-2, dim2=-1)
    if lengths is not None:
        lap += det_offset

    if multi_root:
        rss = torch.diagonal(input, 0, -2, -1).exp()  # root selection scores
        lap = lap + torch.diag_embed(rss, offset=0, dim1=-2, dim2=-1)
        inv_laplacian = lap.inverse()
        factor = (
            torch.diagonal(inv_laplacian, 0, -2, -1)
            .unsqueeze(2)
            .expand_as(input)
            .transpose(1, 2)
        )
        term1 = input.exp().mul(factor).clone()
        term2 = input.exp().mul(inv_laplacian.transpose(1, 2)).clone()
        output = term1 - term2
        roots_output = (
            torch.diagonal(input, 0, -2, -1)
            .exp()
            .mul(torch.diagonal(inv_laplacian.transpose(1, 2), 0, -2, -1))
        )
    else:
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
            torch.diagonal(input, 0, -2, -1)
            .exp()
            .mul(inv_laplacian.transpose(1, 2)[:, 0])
        )
    output = output + torch.diag_embed(roots_output, 0, -2, -1)
    return output
