from .distributions import LinearChainCRF
import torch
from hypothesis import given, settings
from hypothesis.strategies import integers, data, sampled_from

smint = integers(min_value=2, max_value=4)
tint = integers(min_value=1, max_value=2)
lint = integers(min_value=2, max_value=10)


@given(data(), integers(min_value=1, max_value=20))
@settings(max_examples=50, deadline=None)
def test_simple(data, seed):

    model = data.draw(sampled_from([LinearChainCRF]))
    struct = model.struct
    torch.manual_seed(seed)
    vals, (batch, N) = struct._rand()
    lengths = torch.tensor(
        [data.draw(integers(min_value=2, max_value=N)) for b in range(batch - 1)] + [N]
    )

    dist = model(vals, lengths)
    edges, enum_lengths = dist.enumerate_support()
    print(edges.shape)
    log_probs = dist.log_prob(edges)
    for b in range(lengths.shape[0]):
        log_probs[enum_lengths[b] :, b] = -1e9

    assert torch.isclose(log_probs.exp().sum(0), torch.tensor(1.0)).all()

    entropy = dist.entropy
    assert torch.isclose(entropy, -log_probs.exp().mul(log_probs).sum(0)).all()

    argmax = dist.argmax
    _, max_indices = log_probs.max(0)

    amax = edges[max_indices, torch.arange(batch)]
    print(argmax.nonzero())
    print((amax - argmax).nonzero(), lengths)
    assert (amax == argmax).all()

    samples = dist.sample((100,))
    marginals = dist.marginals
    assert ((samples.mean(0) - marginals).abs() < 0.2).all()
