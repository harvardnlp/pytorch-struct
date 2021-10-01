import torch
import torch_struct
import warnings


def test_alignment_crf():
    batch, N, M = 1, 4, 5
    log_potentials = torch.rand(batch, N, M, 3).cuda()

    try:
        log_potentials = log_potentials.cuda()
        on_cuda = True

    except RuntimeError:
        warnings.warn('Could not move log potentials to CUDA device. '
                      'Will not test marginals.')
        on_cuda = False

    dist = torch_struct.AlignmentCRF(log_potentials)
    assert (N, M, 3) == dist.argmax[0].shape
    if on_cuda:
        assert (N, M, 3) == dist.marginals[0].shape
