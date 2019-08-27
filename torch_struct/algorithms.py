import torch
from .semirings import LogSemiring
import opt_einsum as oe

"""
Conventions:
 > N -> length
 > C -> states
"""

def _make_chart(size, potentials, semiring):
    return torch.zeros(*size).type_as(potentials).fill_(semiring.zero())

def linearchain_inside(edge, semiring=LogSemiring):
    """
    Compute forward algorithm on a lattice.

    Parameters:
       edge: a batch x len  x states x states tensor .

    Returns:
       v : sum of all paths
       forward : forward chart
    """
    batch, N, C, _ = edge.shape
    alpha = _make_chart((batch, N+1, C), edge, semiring)
    alpha[:, 0].fill_(semiring.one())
    for n in range(1, N+1):
        alpha[:, n] = semiring.contract("ab,abc->ac",
                                        alpha[:, n-1],
                                        edge[:, n-1])
    return alpha

def linearchain(edge, semiring=LogSemiring):
    alpha = linearchain_inside(edge, semiring)
    return alpha[:, -1].sum()

# Constants
A, B, R, C, L, I = 0, 1, 1, 1, 0, 0
def dependencytree_inside(arcs, semiring=None):
    input = convert(input)
    DIRS = 2
    dot = lambda term, vals: oe.contract(term, *vals, backend=semiring.name)
    stack = lambda a, b: torch.stack([a, b])
    sstack = lambda a: torch.stack([a, a])
    alpha = [ [_make_chart((DIRS, batch_size, n, n), arcs, semiring)
               for _ in [I, C]] for _ in range(2)]
    arcs = [_maketorch.zeros(DIRS, batch_size, n) for _ in range(n)]
    ret = torch.zeros(batch_size, n, n).cpu()

    batch_size, N, _ = arcs.shape

    # Inside step. assumes first token is root symbol
    alpha[A][C][:, :, :, 0].data.fill_(semiring.one())
    alpha[B][C][:, :, :, -1].data.fill_(semiring.one())

    for k in range(1, N):
        f = torch.arange(N-k), torch.arange(k, N)
        arcs[k] = semiring.contract("abcd,abcd,abcd->abc",
                                    sstack(alpha[A][C][R, :, :N-k, :k]),
                                    sstack(alpha[B][C][L, :, k:, N-k:]),
                                    stack(input[:, f[1], f[0]],
                                          input[:, f[0], f[1]]))
        alpha[A][I][:, :, :N-k, k] = arcs[k]
        alpha[B][I][:, :, k:N, N-k-1] = alpha[B][I][:, :, :N-k, k]

        alpha[A][C][: , :, :N-k, k] = \
                                      semiring.contract("abc,abc->ab",
                                                        stack(alpha[A][C][L, :, :N-k, :k],
                                                              alpha[A][I][R, :, :N-k, 1:k+1]),
                                                        stack(alpha[B][I][L, :, k:, N-k-1:N-1],
                                                              alpha[B][C][R, :, k:, N-k:]))
        alpha[B][C][:, :, k:n, N-k-1] = alpha[A][C][ :, :, :N-k, k]
    return arcs

    # # Backward.
    # inputs = []
    # inarcs = []
    # for r in [L, R]:
    #     for k in range(1, n):
    #         if not (r == L and k == n-1):
    #             inputs.append(arcs[k])
    #             inarcs.append(k)
    # v = alpha[C][0, R, :, 0, n-1]

    # grads = torch.autograd.grad(v.sum(), inputs, create_graph=True,
    #                             only_inputs=True, allow_unused=False)


    # for k, grad in zip(inarcs, grads):
    #     f = torch.arange(n-k), torch.arange(k, n)
    #     self.ret[:, f[0], f[1]] = grad[R].cpu()
    #     self.ret[:, f[1], f[0]] = grad[L].cpu()

    # return unconvert(self.ret).cuda()


def dependencytree_nonproj_marginals(arcs, eps=1e-5):
    laplacian = input.exp() + self.eps
    output = input.clone()
    for b in range(input.size(0)):
        lap = laplacian[b].masked_fill(
            torch.eye(input.size(1), device=input.device) != 0, 0)
        lap = -lap + torch.diag(lap.sum(0))
        # store roots on diagonal
        lap[0] = input[b].diag().exp()
        inv_laplacian = lap.inverse()
        factor = inv_laplacian.diag().unsqueeze(1)\
                                     .expand_as(input[b]).transpose(0, 1)
        term1 = input[b].exp().mul(factor).clone()
        term2 = input[b].exp().mul(inv_laplacian.transpose(0, 1)).clone()
        term1[:, 0] = 0
        term2[0] = 0
        output[b] = term1 - term2
        roots_output = input[b].diag().exp().mul(
            inv_laplacian.transpose(0, 1)[0])
        output[b] = output[b] + torch.diag(roots_output)
    return output


def cky(terms, rules, roots, semiring=None):
    #inside step
    #unary scores : b x n x T
    #rule scores : b x NT    x (NT+T) x (NT+T)
    #root : b x NT
    batch_size = unary_scores.size(0)
    n = unary_scores.size(1)
    self.beta = unary_scores.new(batch_size, n, n, self.states).fill_(-self.huge).type_as(unary_scores)
    self.betarev = unary_scores.new(batch_size, n, n, self.states).fill_(-self.huge).type_as(unary_scores)

    beta = self.beta
    betarev = self.betarev

    for state in range(self.t_states):
        beta[:, :, 0, self.nt_states + state] = unary_scores[:, :, state]
        betarev[:, :, n-1, self.nt_states + state] = unary_scores[:, :, state]

    NT = self.nt_states
    S = self.states
    for w in np.arange(1, n):
        B = beta[:, :n-w, :w, :].view(batch_size, n-w, w, 1, S, 1)
        C = betarev[:, w:, n-w:, :] .view(batch_size, n-w, w, 1, 1, S)
        A_B_C = rule_scores[:, :, :, :].view(batch_size, 1,  NT, S, S)
        rules = (self.logsumexp(B + C, dim=2) + A_B_C).view(batch_size, n-w, NT, S*S)
        beta[:, :n-w, w, :NT] = self.logsumexp(rules, dim=3)
        betarev[:, w:n, n-w-1, :NT] = beta[:, :n-w, w, :NT]

    log_Z = self.beta[:, 0, n-1, :self.nt_states] + root_scores
    log_Z = self.logsumexp(log_Z, 1)
    return log_Z
