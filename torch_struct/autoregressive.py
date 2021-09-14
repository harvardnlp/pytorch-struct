import torch
from .semirings import MaxSemiring, KMaxSemiring, TempMax
from torch.distributions.distribution import Distribution


class AutoregressiveModel(torch.nn.Module):
    """
    User should implement as their favorite RNN / Transformer / etc.
    """

    def forward(self, inputs, state=None):
        r"""
        Compute the logits for all tokens in a batched sequence :math:`p(y_{t+1}, ... y_{T}| y_1 \ldots t)`

        Parameters:
            inputs (batch_size x N x C ): next tokens to update representation
            state (tuple of batch_size x ...): everything needed for conditioning.

        Retuns:
            logits (*batch_size x C*): next set of logits.

            state (*tuple of batch_size x ...*): next set of logits.
        """
        pass


def wrap(state, ssize):
    return state.contiguous().view(ssize, -1, *state.shape[1:])


def unwrap(state):
    return state.contiguous().view(-1, *state.shape[2:])


class Autoregressive(Distribution):
    """
    Autoregressive sequence model utilizing beam search.

    * batch_shape -> Given by initializer
    * event_shape -> N x T sequence of choices

    Parameters:
        model (AutoregressiveModel): A lazily computed autoregressive model.
        init (tuple of tensors, batch_shape x ...): initial state of autoregressive model.
        n_classes (int): number of classes in each time step
        n_length (int): max length of sequence
    """

    def __init__(
        self,
        model,
        initial_state,
        n_classes,
        n_length,
        normalize=True,
        start_class=0,
        end_class=None,
    ):
        self.model = model
        self.init = initial_state
        self.n_length = n_length
        self.n_classes = n_classes
        self.start_class = start_class
        self.normalize = normalize
        event_shape = (n_length, n_classes)
        batch_shape = initial_state[0].shape[:1]
        self.device = initial_state[0].device
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def log_prob(self, value, sparse=False):
        """
        Compute log probability over values :math:`p(z)`.

        Parameters:
            value (tensor): One-hot events (*sample_shape x batch_shape x N*)
            sparse (bool): implement sparse

        Returns:
            log_probs (*sample_shape x batch_shape*)
        """
        value = value.long()
        if not sparse:
            sample, batch_shape, n_length, n_classes = value.shape
            value = (
                (value * torch.arange(n_classes).view(1, 1, n_classes)).sum(-1).long()
            )
        else:
            sample, batch_shape, n_length = value.shape

        value = torch.cat(
            [
                torch.zeros(sample, batch_shape, 1, device=value.device)
                .fill_(self.start_class)
                .long(),
                value,
            ],
            dim=2,
        )
        value = unwrap(value)
        state = tuple(
            (unwrap(i.unsqueeze(0).expand((sample,) + i.shape)) for i in self.init)
        )

        logits, _ = self.model(value, state)
        b2, n2, c2 = logits.shape
        assert (
            (b2 == sample * batch_shape)
            and (n2 == n_length + 1)
            and (c2 == self.n_classes)
        ), "Model should return logits of shape `batch x N x C` "

        if self.normalize:
            log_probs = logits.log_softmax(-1)
        else:
            log_probs = logits

        scores = log_probs[:, :-1].gather(2, value[:, 1:].unsqueeze(-1)).sum(-1).sum(-1)
        return wrap(scores, sample)

    def _beam_search(self, semiring, gumbel=False):
        beam = semiring.fill(
            torch.zeros((semiring.size(),) + self.batch_shape, device=self.device),
            torch.tensor(True),
            semiring.one,
        )
        ssize = semiring.size()

        def take(state, indices):
            return tuple(
                (
                    s.contiguous()[
                        (
                            indices * self.batch_shape[0]
                            + torch.arange(
                                self.batch_shape[0], device=self.device
                            ).unsqueeze(0)
                        )
                        .contiguous()
                        .view(-1)
                    ]
                    for s in state
                )
            )

        tokens = (
            torch.zeros((ssize * self.batch_shape[0])).long().fill_(self.start_class)
        )
        state = tuple(
            (unwrap(i.unsqueeze(0).expand((ssize,) + i.shape)) for i in self.init)
        )

        # Beam Search
        all_beams = []
        all_logits = []
        for t in range(0, self.n_length):
            logits, state = self.model(unwrap(tokens).unsqueeze(1), state)
            b2, n2, c2 = logits.shape
            assert (
                (b2 == ssize * self.batch_shape[0])
                and (n2 == 1)
                and (c2 == self.n_classes)
            ), "Model should return logits of shape `batch x N x C` "
            for s in state:
                assert (
                    s.shape[0] == ssize * self.batch_shape[0]
                ), "Model should return state tuple with shapes `batch x ...` "
            logits = wrap(logits.squeeze(1), ssize)
            if gumbel:
                logits = logits + torch.distributions.Gumbel(0.0, 1.0).sample(
                    logits.shape
                )
            if self.normalize:
                logits = logits.log_softmax(-1)
            all_logits.append(logits)
            ex_beam = beam.unsqueeze(-1) + logits
            ex_beam.requires_grad_(True)
            all_beams.append(ex_beam)
            beam, (positions, tokens) = semiring.sparse_sum(ex_beam)
            state = take(state, positions)

        # Back pointers
        v = beam
        all_m = []
        for k in range(v.shape[0]):
            obj = v[k].sum(dim=0)
            marg = torch.autograd.grad(
                obj, all_beams, create_graph=True, only_inputs=True, allow_unused=False
            )
            marg = torch.stack(marg, dim=2)
            all_m.append(marg.sum(0))
        return torch.stack(all_m, dim=0), v, torch.stack(all_logits, dim=2)

    def greedy_max(self):
        """
        Compute "argmax" using greedy search.

        Returns:
            greedy_path (*batch x N x C*)
            greedy_max (*batch*)
            logits (*batch x N x C*)
        """
        a, b, c = self._beam_search(MaxSemiring)
        return a.squeeze(0), b.squeeze(0), c.squeeze(0)

    def greedy_tempmax(self, alpha):
        """
        Compute differentiable scheduled sampling using greedy search.

        Based on:

        * Differentiable Scheduled Sampling for Credit Assignment :cite:`goyal2017differentiable`

        Parameters:
            alpha : alpha param

        Returns:
            greedy_path (*batch x N x C*)
            greedy_max (*batch*)
            logits (*batch x N x C*)
        """
        a, b, c = self._beam_search(TempMax(alpha), alpha)
        return a.squeeze(0), b.squeeze(0), c.squeeze(0)

    def beam_topk(self, K):
        """
        Compute "top-k" using beam search

        Parameters:
            K : top-k

        Returns:
            paths (*K x batch x N x C*)

        """
        return self._beam_search(KMaxSemiring(K))[0]

    def _beam_max(self, K):
        return self._beam_search(KMaxSemiring(K))[1]

    def sample_without_replacement(self, sample_shape=torch.Size()):
        """
        Compute sampling without replacement using Gumbel trick.

        Based on:

        * Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for
               Sampling Sequences Without Replacement :cite:`DBLP:journals/corr/abs-1903-06059`

        Parameters:
            sample_shape (torch.Size): batch_size

        Returns:
            paths (*K x batch x N x C*)

        """
        K = sample_shape[0]
        return self._beam_search(KMaxSemiring(K), gumbel=True)[0]

    def sample(self, sample_shape=torch.Size()):
        r"""
        Compute structured samples from the distribution :math:`z \sim p(z)`.

        Parameters:
            sample_shape (torch.Size): number of samples

        Returns:
            samples (*sample_shape x batch_shape x event_shape*)
        """
        sample_shape = sample_shape[0]
        state = tuple(
            (
                unwrap(i.unsqueeze(0).expand((sample_shape,) + i.shape))
                for i in self.init
            )
        )
        all_tokens = []
        tokens = (
            torch.zeros((sample_shape * self.batch_shape[0]))
            .long()
            .fill_(self.start_class)
        )
        for t in range(0, self.n_length):
            logits, state = self.model(tokens.unsqueeze(-1), state)
            logits = logits.squeeze(1)
            tokens = torch.distributions.Categorical(logits=logits).sample((1,))[0]
            all_tokens.append(tokens)
        v = wrap(torch.stack(all_tokens, dim=1), sample_shape)
        return torch.nn.functional.one_hot(v, self.n_classes)
