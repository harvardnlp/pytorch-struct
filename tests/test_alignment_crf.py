import torch
import torch_struct
import warnings


def test_alignment_crf():
    batch, N, M = 1, 4, 5
    log_potentials = torch.rand(batch, N, M, 3)

    if torch.cuda.is_available():
        log_potentials = log_potentials.cuda()
    else:
        warnings.warn('Could not move log potentials to CUDA device. '
                      'Will not test marginals.')

    dist = torch_struct.AlignmentCRF(log_potentials)
    assert (N, M, 3) == dist.argmax[0].shape
    if torch.cuda.is_available():
        assert (N, M, 3) == dist.marginals[0].shape
