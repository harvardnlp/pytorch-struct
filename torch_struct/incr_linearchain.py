
import torch
from torch_struct.helpers import _Struct
from einops import rearrange, repeat
import math

class IncrementLinearChainFwBw(_Struct):

    def _check_potentials(self, edge, lengths=None):
        batch, N_1, C, C2 = self._get_dimension(edge)
        edge = self.semiring.convert(edge)
        N = N_1 + 1

        if lengths is None:
            lengths = torch.LongTensor([N] * batch).to(edge.device)
            # pass
        else:
            assert max(lengths) <= N, "Length longer than edge scores"
            assert max(lengths) == N, "One length must be at least N"
        assert C == C2, "Transition shape doesn't match"
        return edge, batch, N, C, lengths

    def marginals(self, log_potentials, lengths=None, force_grad=True):

        semiring = self.semiring
        ssize = semiring.size()
        log_potentials, batch, N, C, lengths = self._check_potentials(
            log_potentials, lengths
        )
        N1 = N - 1
        log_N, bin_N = self._bin_length(N - 1)
        chart = self._chart((batch, N1, C, C), log_potentials, force_grad)
        #print(chart.size())
        
        init = torch.zeros(ssize, batch, N1, C, C).bool().to(log_potentials.device)
        init.diagonal(0, -2, -1).fill_(True)
        chart = semiring.fill(chart, init, semiring.one)

        big = torch.zeros(
            ssize, 
            batch,
            N1, 
            C,
            C,
            dtype=log_potentials.dtype,
            device=log_potentials.device,
        )
        big[:, :, : N1] = log_potentials
        c = chart[:, :, :].view(ssize, batch * N1, C, C)
        lp = big[:, :, :].view(ssize, batch * N1, C, C)
        mask = torch.arange(N1).view(1, N1).expand(batch, N1).type_as(c)
        mask = mask >= (lengths - 1).view(batch, 1)
        mask = mask.view(batch * N1, 1, 1).to(lp.device)
        lp.data[:] = semiring.fill(lp.data, mask, semiring.zero)
        c.data[:] = semiring.fill(c.data, ~mask, semiring.zero)

        c[:] = semiring.sum(torch.stack([c.data, lp], dim=-1))
        #print(chart)
        
        # initialize interval chart
        interval_chart = self._chart((batch, N1, N1, C, C), log_potentials, force_grad)
        init = torch.zeros(ssize, batch, N1, N1, C, C).bool().to(log_potentials.device)
        init.diagonal(0, -2, -1).fill_(True)
        interval_chart = semiring.fill(interval_chart, init, semiring.one)
        
        # first row of interval chart
        interval_chart[:, :, 0] = chart
        
        #print(interval_chart.size())
        #print(interval_chart[:, :, 0])
        
        for n in range(0, log_N):
            
            index = int(math.pow(2, n))
            
            # special case for the last iteration
            height = min(index, N1 - index)
            
            l2 = interval_chart[:, :, : height, index :].contiguous()
            l1 = interval_chart[:, :, index - 1, : N1 - index].unsqueeze(2).expand_as(l2).contiguous()
            #assert(l1.shape == l2.shape), (l1.shape, l2.shape)
            #print(f"n: {n}, N1: {N1}, logN: {log_N}")
            #print(l2.size())
            #rint(l1.size())
            _l2 = l2.view(-1, l2.size(-2), l2.size(-1))
            _l1 = l1.view(-1, l1.size(-2), l1.size(-1))
            _l_update = semiring.matmul(_l2, _l1)#.contiguous()#.clone(memory_format = torch.contiguous_format)
            #print(_l_update.size())
            l_uptate = _l_update.view(*l2.shape)
            #fill_mask = torch.zeros()
            interval_chart[:, :, index : index + height, : N1 - index] = l_uptate
            #interval_chart = interval_chart.contiguous()
            
        #print(semiring.sum(semiring.sum(interval_chart)))
        
        
        # calculate marginal using fw-bw property
        # p(z_i=c | x_1:t) = exp( l_1:i(c,.) + l_i+1:t(.,c) - l_1:t(.,.) ) = exp( L_prefix + L_interval - L_total)
        
        #mask_value = - float("inf")
        tril_mask = torch.tril(torch.ones(N1, N1)).to(interval_chart.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        
        L_p = semiring.sum(interval_chart[:, :, :, 0].contiguous(), dim=-1).unsqueeze(2).expand(*interval_chart.shape[:-1])
        #print(L_p.size(), tril_mask.size())
        L_p = L_p * tril_mask.float()
        #print(L_p)
        
        L_i = semiring.sum(interval_chart, dim=-2)
        index_matrix = torch.stack([torch.arange(-i, N1 - i) % (N1) for i in range(N1)], dim=1).to(L_i.device)
        index_matrix = repeat(index_matrix, "i j -> () () i j ()").expand_as(L_i)
        # re-index into tril
        #print(L_i)
        L_i = torch.gather(L_i, 2, index_matrix)
        L_i = L_i * tril_mask.float()
        # shift
        L_i_final = torch.zeros_like(L_p, device=L_p.device)
        L_i_final[:, :, :, :-1] = L_i[:, :, :, 1:]
        #print(L_i_final)
        
        L_t = semiring.sum(semiring.sum(interval_chart[:, :, :, 0].contiguous())).unsqueeze(3).expand(*interval_chart.shape[:-2])
        # don't mask denominator
        #print(L_t)
        
        #print(L_p.size(), L_i_final.size(), L_t.size())
        return ((L_p + L_i_final - L_t.unsqueeze(-1)).exp() * tril_mask.float()).squeeze(0)