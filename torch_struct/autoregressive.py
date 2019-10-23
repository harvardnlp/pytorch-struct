import torch
from .semirings import MaxSemiring
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property

class Autoregressive(Distribution):
    """
    Autoregressive sequence model utilizing beam search.

    * batch_shape -> Given by initializer
    * event_shape -> N x T sequence of choices

    Parameters:
        model:
        init (tensor, batch_shape x hidden_shape):
        n_classes (int): number of classes in each time step
        n_length (int): max length of sequence

    """

    def __init__(self, model, init, n_classes, n_length):
        self.model = model
        self.init = init
        self.n_length = n_length
        self.n_classes = n_classes
        event_shape = (n_length, n_classes)
        batch_shape = start_state.shape[:1]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)


    def log_prob(self, value):
        """
        Compute log probability over values :math:`p(z)`.

        Parameters:
            value (tensor): One-hot events (*sample_shape x batch_shape x event_shape*)

        Returns:
            log_probs (*sample_shape x batch_shape*)
        """
        logits = self.model.sequence_logits(self.init, value)
        # batch_shape x event_shape (N x C)
        log_probs = logits.log_softmax(-1)
        positions = torch.arange(self.n_length)
        return log_probs[:, positions, value[positions]].sum(-1)

    def _beam_search(self, semiring):
        # beam size
        beam = semiring.ones_(
            torch.zeros(semiring.size(), self.batch_shape))
        state = self.init.unsqueeze(0).expand((semiring.size(),) + self.init.shape)
        for t in range(0, self.n_length):
            logits = self.model.logits(init)
            # ssize x batch_size x C
            beam = semiring.times(beam.unsqueeze(-1), logits.log_softmax(-1))
            # ssize x batch_size x C
            beam, backpointers = semiring.sparse_sum(beam)
            # ssize x batch_size
            state = self.model.update_state(state, backpointers)
        return beam

    def greedy_max(self):
        return _beam_search(self, MaxSemiring)

    def sample(self, sample_shape=torch.Size()):
        r"""
        Compute structured samples from the distribution :math:`z \sim p(z)`.

        Parameters:
            sample_shape (int): number of samples

        Returns:
            samples (*sample_shape x batch_shape x event_shape*)
        """
        pass
