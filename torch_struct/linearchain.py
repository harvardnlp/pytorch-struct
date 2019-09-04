import torch
from .helpers import _Struct


class LinearChain(_Struct):
    def _dp(self, edge, lengths=None, force_grad=False):
        semiring = self.semiring
        batch, N_1, C, C2 = edge.shape
        N = N_1 + 1
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)
        assert max(lengths) <= N, "Length longer than edge scores"
        assert max(lengths) == N, "One length must be at least N"
        assert C == C2, "Transition shape doesn't match"

        alpha = self._make_chart(N, (batch, C), edge, force_grad=force_grad)

        edge_store = self._make_chart(N - 1, (batch, C, C), edge, force_grad=force_grad)

        alpha[0].data.fill_(semiring.one())
        for n in range(1, N):
            edge_store[n - 1][:] = semiring.times(
                alpha[n - 1].view(batch, 1, C), edge[:, n - 1]
            )
            alpha[n][:] = semiring.sum(edge_store[n - 1])
        v = semiring.sum(
            torch.stack([alpha[l - 1][i] for i, l in enumerate(lengths)]), dim=-1
        )
        return v, edge_store

    def sum(self, edge, lengths=None):
        """
        Compute the forward pass of a linear chain CRF.

        Parameters:
            edge : b x (N-1) x C x C markov potentials
                        (n-1 x z_n x z_{n-1})
            lengths: None or b long tensor mask

        Returns:
            v: b tensor of total sum
            inside: list of N,  b x C x C table

        """
        return self._dp(edge, lengths)[0]

    def marginals(self, edge, lengths=None):
        """
        Compute the marginals of a linear chain CRF.

        Parameters:
            edge : b x (N-1) x C x C markov potentials
                        (t x z_t x z_{t-1})
            lengths: None or b long tensor mask
        Returns:
            marginals: b x (N-1) x C x C table

        """
        v, alpha = self._dp(edge, lengths=lengths, force_grad=True)
        marg = torch.autograd.grad(
            v.sum(dim=0), alpha, create_graph=True, only_inputs=True, allow_unused=False
        )
        return torch.stack(marg, dim=1)

    # Adapters
    @staticmethod
    def hmm(transition, emission, init, observations):
        """
        Convert HMM to a linear chain.

        Parameters:
            transition: C X C
            emission: V x C
            init: C
            observations: b x N between [0, V-1]

        Returns:
            edges: b x (N-1) x C x C
        """
        V, C = emission.shape
        batch, N = observations.shape
        scores = torch.ones(batch, N - 1, C, C).type_as(emission)
        scores[:, :, :, :] *= transition.view(1, 1, C, C)
        scores[:, 0, :, :] *= init.view(1, 1, C)
        obs = emission[observations.view(batch * N), :]
        scores[:, :, :, :] *= obs.view(batch, N, C, 1)[:, 1:]
        scores[:, 0, :, :] *= obs.view(batch, N, 1, C)[:, 0]
        return scores

    @staticmethod
    def to_parts(sequence, extra, lengths=None):
        """
        Convert a sequence representation to edges

        Parameters:
            sequence : b x N long tensor in [0, C-1]
            C : number of states
            lengths: b long tensor of N values
        Returns:
            edge : b x (N-1) x C x C markov indicators
                        (t x z_t x z_{t-1})
        """
        C = extra
        batch, N = sequence.shape
        labels = torch.zeros(batch, N - 1, C, C).long()
        if lengths is None:
            lengths = torch.LongTensor([N] * batch)
        for n in range(1, N):
            labels[torch.arange(batch), n - 1, sequence[:, n], sequence[:, n - 1]] = 1
        for b in range(batch):
            labels[b, lengths[b] - 1 :, :, :] = 0
        return labels

    @staticmethod
    def from_parts(edge):
        """
        Convert edges to sequence representation.

        Parameters:
            edge : b x (N-1) x C x C markov indicators
                        (t x z_t x z_{t-1})
        Returns:
            sequence : b x N long tensor in [0, C-1]
        """
        batch, N_1, C, _ = edge.shape
        N = N_1 + 1
        labels = torch.zeros(batch, N).long()
        on = edge.nonzero()
        for i in range(on.shape[0]):
            if on[i][1] == 0:
                labels[on[i][0], on[i][1]] = on[i][3]
            labels[on[i][0], on[i][1] + 1] = on[i][2]
        return labels, C

    @staticmethod
    def _rand():
        b = torch.randint(2, 4, (1,))
        N = torch.randint(2, 4, (1,))
        C = torch.randint(2, 4, (1,))
        return torch.rand(b, N, C, C), (b.item(), (N + 1).item())

    ### Tests
    def enumerate(self, edge):
        semiring = self.semiring
        batch, N, C, _ = edge.shape
        chains = [([c], torch.zeros(batch).fill_(semiring.one())) for c in range(C)]
        for n in range(1, N + 1):
            new_chains = []
            for chain, score in chains:
                for c in range(C):
                    new_chains.append(
                        (chain + [c], semiring.mul(score, edge[:, n - 1, c, chain[-1]]))
                    )
            chains = new_chains

        edges = self.to_parts(torch.stack([torch.tensor(c) for (c, _) in chains]), C)
        a = (
            torch.einsum("ancd,bncd->bancd", edges.float(), edge)
            .sum(dim=2)
            .sum(dim=2)
            .sum(dim=2)
        )
        a = semiring.sum(a, dim=1)
        b = semiring.sum(torch.stack([s for (_, s) in chains]), dim=0)
        assert torch.isclose(a, b).all()
        return semiring.sum(torch.stack([s for (_, s) in chains]), dim=0)
