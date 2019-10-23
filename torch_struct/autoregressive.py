import torch
from .semirings import MaxSemiring, KMaxSemiring
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property


class AutoregressiveModel:
    """
    User should implement as their favorite RNN / Transformer / etc.
    """

    def sequence_logits(self, init, seq_inputs):
        """
        Compute the logits for all tokens in a batched sequence :math:`p(y_1, ... y_{T})`

        Parameters:
            init (batch_size x hidden_shape): everything needed for conditioning.
            inputs (batch_size x N x C): next tokens to update representation

        Retuns:
            logits (batch_size x C): next set of logits.
        """
        pass

    def local_logits(self, state):
        """
        Compute the local logits of :math:`p(y_t | y_{1:t-1})`

        Parameters:
            state (batch_size x hidden_shape): everything needed for conditioning.

        Retuns:
            logits (batch_size x C): next set of logits.
        """
        pass

    def update_state(self, prev_state, inputs):
        """
        Update the model state based on previous state and inputs

        Parameters:
            prev_state (batch_size x hidden_shape): everything needed for conditioning.
            inputs (batch_size x C): next tokens to update representation

        Retuns:
            state (batch_size x hidden_shape): everything needed for next conditioning.
        """
        pass


class Autoregressive(Distribution):
    """
    Autoregressive sequence model utilizing beam search.

    * batch_shape -> Given by initializer
    * event_shape -> N x T sequence of choices

    Parameters:
        model (AutoregressiveModel): A lazily computed autoregressive model.
        init (tensor, batch_shape x hidden_shape): initial state of autoregressive model.
        n_classes (int): number of classes in each time step
        n_length (int): max length of sequence

    """

    def __init__(self, model, init, n_classes, n_length):
        self.model = model
        self.init = init
        self.n_length = n_length
        self.n_classes = n_classes
        event_shape = (n_length, n_classes)
        batch_shape = init.shape[:1]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def log_prob(self, value, normalize=True):
        """
        Compute log probability over values :math:`p(z)`.

        Parameters:
            value (tensor): One-hot events (*sample_shape x batch_shape x event_shape*)

        Returns:
            log_probs (*sample_shape x batch_shape*)
        """
        batch_shape, n_length, n_classes = value.shape
        value = value.long()
        logits = self.model.sequence_logits(self.init, value)
        if normalize:
            log_probs = logits.log_softmax(-1)
        else:
            log_probs = logits

        # batch_shape x event_shape (N x C)
        positions = torch.arange(self.n_length)
        batch = torch.arange(batch_shape)
        return log_probs.masked_fill_(value == 0, 0).sum(-1).sum(-1)

    def _beam_search(self, semiring, gumbel=True):
        beam = semiring.one_(torch.zeros((semiring.size(),) + self.batch_shape))
        state = self.init.unsqueeze(0).expand((semiring.size(),) + self.init.shape)

        # Beam Search
        all_beams = []
        for t in range(0, self.n_length):
            logits = self.model.local_logits(state)
            if gumbel:
                logits = logits + torch.distributions.Gumbel(0.0, 0.0).sample(
                    logits.shape
                )

            ex_beam = beam.unsqueeze(-1) + logits
            ex_beam.requires_grad_(True)
            all_beams.append(ex_beam)
            beam, tokens = semiring.sparse_sum(ex_beam)
            state = self.model.update_state(state, tokens)

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
        return torch.stack(all_m, dim=0)

    def greedy_argmax(self):
        """
        Compute "argmax" using greedy search
        """
        return self._beam_search(MaxSemiring).squeeze(0)

    def beam_topk(self, K):
        """
        Compute "top-k" using beam search
        """
        return self._beam_search(KMaxSemiring(K))

    def sample_without_replacement(self, sample_shape=torch.Size()):
        """
        Compute sampling without replacement using Gumbel trick.
        """
        K = sample_shape[0]
        return self._beam_search(KMaxSemiring(K), gumbel=True)

    def sample(self, sample_shape=torch.Size()):
        r"""
        Compute structured samples from the distribution :math:`z \sim p(z)`.

        Parameters:
            sample_shape (int): number of samples

        Returns:
            samples (*sample_shape x batch_shape x event_shape*)
        """
        sample_shape = sample_shape[0]
        beam = torch.zeros((sample_shape,) + self.batch_shape)
        state = self.init.unsqueeze(0).expand((sample_shape,) + self.init.shape)
        all_tokens = []
        for t in range(0, self.n_length):
            logits = self.model.local_logits(state)
            tokens = torch.distributions.OneHotCategorical(logits).sample((1,))[0]
            state = self.model.update_state(state, tokens)
            all_tokens.append(tokens)
        return torch.stack(all_tokens, dim=2)
