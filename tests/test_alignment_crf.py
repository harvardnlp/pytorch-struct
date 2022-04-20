import torch
import torch_struct
import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_alignment_crf_shapes():
    batch, N, M = 2, 4, 5
    log_potentials = torch.rand(batch, N, M, 3).cuda()

    dist = torch_struct.AlignmentCRF(log_potentials)
    assert (batch, N, M, 3) == dist.argmax.shape
    assert (batch, N, M, 3) == dist.marginals.shape
    assert (batch,) == dist.partition.shape

    # Fail due to AttributeError: 'BandedMatrix' object has no attribute
    # 'unsqueeze'
    assert (batch,) == dist.entropy.shape
    # assert (9, batch, N, M, 3) == dist.sample([9]).shape

    # Fails due to: RuntimeError: Expected condition, x and y to be on
    # the same device, but condition is on cpu and x and y are on
    # cuda:0 and cuda:0 respectively
    # assert (8, batch,) == dist.topk(8).shape
