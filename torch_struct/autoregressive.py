import torch
from .semirings import MaxSemiring, KMaxSemiring
from torch.distributions.distribution import Distribution


class AutoregressiveModel(torch.nn.Module):
    """
    User should implement as their favorite RNN / Transformer / etc.
    """

    def forward(self, inputs, state=None):
        r"""
        Compute the logits for all tokens in a batched sequence :math:`p(y_{t+1}, ... y_{T}| y_1 \ldots t)`

        Parameters:
            inputs (batch_size x N x C): next tokens to update representation
            state (tuple of batch_size x ...): everything needed for conditioning.

        Retuns:
            logits (batch_size x C): next set of logits.
            state (tuple of batch_size x ...): next set of logits.
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

    def __init__(self, model, init, n_classes, n_length):
        self.model = model
        self.init = init
        self.n_length = n_length
        self.n_classes = n_classes
        event_shape = (n_length, n_classes)
        batch_shape = init[0].shape[:1]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def log_prob(self, value, normalize=True):
        """
        Compute log probability over values :math:`p(z)`.

        Parameters:
            value (tensor): One-hot events (*sample_shape x batch_shape x N*)

        Returns:
            log_probs (*sample_shape x batch_shape*)
        """
        n_length, batch_shape, n_classes = value.shape
        value = value.long()
        logits, _ = self.model(value, self.init)
        if normalize:
            log_probs = logits.log_softmax(-1)
        else:
            log_probs = logits

        # batch_shape x event_shape (N x C)
        return log_probs.masked_fill_(value == 0, 0).sum(-1).sum(-1)

    def _beam_search(self, semiring, gumbel=False):
        beam = semiring.one_(torch.zeros((semiring.size(),) + self.batch_shape))
        ssize = semiring.size()

        def take(state, indices):
            return tuple((
                s.contiguous()[(indices +
                                torch.arange(self.batch_shape[0]).unsqueeze(0) * ssize).contiguous().view(-1)]
                for s in state))
        tokens = torch.zeros((ssize * self.batch_shape[0])).long()
        state = tuple((
            unwrap(i.unsqueeze(0).expand((ssize,) + i.shape))
            for i in self.init))

        # Beam Search
        all_beams = []
        for t in range(0, self.n_length):
            logits, state  = self.model(unwrap(tokens).unsqueeze(1),
                                        state)
            logits = wrap(logits.squeeze(1), ssize)
            if gumbel:
                logits = logits + torch.distributions.Gumbel(0.0, 0.0).sample(
                    logits.shape
                )
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
        state = tuple((
            unwrap(i.unsqueeze(0).expand((sample_shape,) + i.shape))
            for i in self.init))
        all_tokens = []
        tokens = torch.zeros((sample_shape * self.batch_shape[0])).long()
        for t in range(0, self.n_length):
            print(tokens.unsqueeze(-1).shape)
            logits, state = self.model(tokens.unsqueeze(-1), state)
            print("l", logits.shape)
            tokens = torch.distributions.Categorical(logits).sample((1,))[0]
            print("t", tokens.shape)
            all_tokens.append(tokens)
        v = wrap(torch.stack(all_tokens, dim=1), sample_shape)
        return torch.nn.functional.one_hot(v, self.n_classes)
