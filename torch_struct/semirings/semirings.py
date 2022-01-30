import torch

has_genbmm = False
try:
    import genbmm

    has_genbmm = True
except ImportError:
    pass


def matmul(cls, a, b):
    dims = 1
    act_on = -(dims + 1)
    a = a.unsqueeze(-1)
    b = b.unsqueeze(act_on - 1)
    c = cls.times(a, b)
    for d in range(act_on, -1, 1):
        c = cls.sum(c.transpose(-2, -1))
    return c


class Semiring:
    """
    Base semiring class.

    Based on description in:

    * Semiring parsing :cite:`goodman1999semiring`

    """

    @classmethod
    def matmul(cls, a, b):
        "Generalized tensordot. Classes should override."
        return matmul(cls, a, b)

    @classmethod
    def size(cls):
        "Additional *ssize* first dimension needed."
        return 1

    @classmethod
    def dot(cls, a, b):
        "Dot product along last dim."
        a = a.unsqueeze(-2)
        b = b.unsqueeze(-1)
        return cls.matmul(a, b).squeeze(-1).squeeze(-1)

    @staticmethod
    def fill(c, mask, v):
        mask = mask.to(c.device)
        return torch.where(
            mask, v.type_as(c).view((-1,) + (1,) * (len(c.shape) - 1)), c
        )

    @classmethod
    def times(cls, *ls):
        "Multiply a list of tensors together"
        cur = ls[0]
        for l in ls[1:]:
            cur = cls.mul(cur, l)
        return cur

    @classmethod
    def convert(cls, potentials):
        "Convert to semiring by adding an extra first dimension."
        return potentials.unsqueeze(0)

    @classmethod
    def unconvert(cls, potentials):
        "Unconvert from semiring by removing extra first dimension."
        return potentials.squeeze(0)

    @staticmethod
    def sum(xs, dim=-1):
        "Sum over *dim* of tensor."
        raise NotImplementedError()

    @classmethod
    def plus(cls, a, b):
        return cls.sum(torch.stack([a, b], dim=-1))


class _Base(Semiring):
    zero = torch.tensor(0.0)
    one = torch.tensor(1.0)

    @staticmethod
    def mul(a, b):
        return torch.mul(a, b)

    @staticmethod
    def prod(a, dim=-1):
        return torch.prod(a, dim=dim)


class _BaseLog(Semiring):
    zero = torch.tensor(-1e5)
    one = torch.tensor(-0.0)

    @staticmethod
    def sum(xs, dim=-1):
        return torch.logsumexp(xs, dim=dim)

    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def prod(a, dim=-1):
        return torch.sum(a, dim=dim)

    # @classmethod
    # def matmul(cls, a, b):
    #     return super(cls).matmul(a, b)


class StdSemiring(_Base):
    """
    Implements the counting semiring (+, *, 0, 1).
    """

    @staticmethod
    def sum(xs, dim=-1):
        return torch.sum(xs, dim=dim)

    @classmethod
    def matmul(cls, a, b):
        "Dot product along last dim"

        if has_genbmm and isinstance(a, genbmm.BandedMatrix):
            return b.multiply(a.transpose())
        else:
            return torch.matmul(a, b)


class LogSemiring(_BaseLog):
    """
    Implements the log-space semiring (logsumexp, +, -inf, 0).

    Gradients give marginals.
    """

    @classmethod
    def matmul(cls, a, b):
        if has_genbmm and isinstance(a, genbmm.BandedMatrix):
            return b.multiply_log(a.transpose())
        else:
            return _BaseLog.matmul(a, b)


class MaxSemiring(_BaseLog):
    """
    Implements the max semiring (max, +, -inf, 0).

    Gradients give argmax.
    """

    @classmethod
    def matmul(cls, a, b):
        if has_genbmm and isinstance(a, genbmm.BandedMatrix):
            return b.multiply_max(a.transpose())
        else:
            return matmul(cls, a, b)

    @staticmethod
    def sum(xs, dim=-1):
        return torch.max(xs, dim=dim)[0]

    @staticmethod
    def sparse_sum(xs, dim=-1):
        m, a = torch.max(xs, dim=dim)
        return m, (torch.zeros(a.shape).long(), a)


def KMaxSemiring(k):
    "Implements the k-max semiring (kmax, +, [-inf, -inf..], [0, -inf, ...])."

    class KMaxSemiring(_BaseLog):

        zero = torch.tensor([-1e5 for i in range(k)])
        one = torch.tensor([0 if i == 0 else -1e5 for i in range(k)])

        @staticmethod
        def size():
            return k

        @classmethod
        def convert(cls, orig_potentials):
            potentials = torch.zeros(
                (k,) + orig_potentials.shape,
                dtype=orig_potentials.dtype,
                device=orig_potentials.device,
            )
            potentials = cls.fill(potentials, torch.tensor(True), cls.zero)
            potentials[0] = orig_potentials
            return potentials

        @staticmethod
        def unconvert(potentials):
            return potentials[0]

        @staticmethod
        def sum(xs, dim=-1):
            if dim == -1:
                xs = xs.permute(tuple(range(1, xs.dim())) + (0,))
                xs = xs.contiguous().view(xs.shape[:-2] + (-1,))
                xs = torch.topk(xs, k, dim=-1)[0]
                xs = xs.permute((xs.dim() - 1,) + tuple(range(0, xs.dim() - 1)))
                assert xs.shape[0] == k
                return xs
            assert False

        @staticmethod
        def sparse_sum(xs, dim=-1):
            if dim == -1:
                xs = xs.permute(tuple(range(1, xs.dim())) + (0,))
                xs = xs.contiguous().view(xs.shape[:-2] + (-1,))
                xs, xs2 = torch.topk(xs, k, dim=-1)
                xs = xs.permute((xs.dim() - 1,) + tuple(range(0, xs.dim() - 1)))
                xs2 = xs2.permute((xs.dim() - 1,) + tuple(range(0, xs.dim() - 1)))
                assert xs.shape[0] == k
                return xs, (xs2 % k, xs2 // k)
            assert False

        @staticmethod
        def mul(a, b):
            a = a.view((k, 1) + a.shape[1:])
            b = b.view((1, k) + b.shape[1:])
            c = a + b
            c = c.contiguous().view((k * k,) + c.shape[2:])
            ret = torch.topk(c, k, 0)[0]
            assert ret.shape[0] == k
            return ret

    return KMaxSemiring


class KLDivergenceSemiring(Semiring):
    """
    Implements an KL-divergence semiring.

    Computes both the log-values of two distributions and the running KL divergence between two distributions.

    Based on descriptions in:

    * Parameter estimation for probabilistic finite-state
      transducers :cite:`eisner2002parameter`
    * First-and second-order expectation semirings with applications to
      minimumrisk training on translation forests :cite:`li2009first`
    * Sample Selection for Statistical Grammar Induction :cite:`hwa2000samplesf`

    """

    zero = torch.tensor([-1e5, -1e5, 0.0])
    one = torch.tensor([0.0, 0.0, 0.0])

    @staticmethod
    def size():
        return 3

    @staticmethod
    def convert(xs):
        values = torch.zeros((3,) + xs[0].shape).type_as(xs[0])
        values[0] = xs[0]
        values[1] = xs[1]
        values[2] = 0
        return values

    @staticmethod
    def unconvert(xs):
        return xs[-1]

    @staticmethod
    def sum(xs, dim=-1):
        assert dim != 0
        d = dim - 1 if dim > 0 else dim
        part_p = torch.logsumexp(xs[0], dim=d)
        part_q = torch.logsumexp(xs[1], dim=d)
        log_sm_p = xs[0] - part_p.unsqueeze(d)
        log_sm_q = xs[1] - part_q.unsqueeze(d)
        sm_p = log_sm_p.exp()
        return torch.stack(
            (
                part_p,
                part_q,
                torch.sum(
                    xs[2].mul(sm_p) - log_sm_q.mul(sm_p) + log_sm_p.mul(sm_p), dim=d
                ),
            )
        )

    @staticmethod
    def mul(a, b):
        return torch.stack((a[0] + b[0], a[1] + b[1], a[2] + b[2]))

    @classmethod
    def prod(cls, xs, dim=-1):
        return xs.sum(dim)


class CrossEntropySemiring(Semiring):
    """
    Implements an cross-entropy expectation semiring.

    Computes both the log-values of two distributions and the running cross entropy between two distributions.

    Based on descriptions in:

    * Parameter estimation for probabilistic finite-state transducers :cite:`eisner2002parameter`
    * First-and second-order expectation semirings with applications to minimum-risk training on translation forests :cite:`li2009first`
    * Sample Selection for Statistical Grammar Induction :cite:`hwa2000samplesf`
    """

    zero = torch.tensor([-1e5, -1e5, 0.0])
    one = torch.tensor([0.0, 0.0, 0.0])

    @staticmethod
    def size():
        return 3

    @staticmethod
    def convert(xs):
        values = torch.zeros((3,) + xs[0].shape).type_as(xs[0])
        values[0] = xs[0]
        values[1] = xs[1]
        values[2] = 0
        return values

    @staticmethod
    def unconvert(xs):
        return xs[-1]

    @staticmethod
    def sum(xs, dim=-1):
        assert dim != 0
        d = dim - 1 if dim > 0 else dim
        part_p = torch.logsumexp(xs[0], dim=d)
        part_q = torch.logsumexp(xs[1], dim=d)
        log_sm_p = xs[0] - part_p.unsqueeze(d)
        log_sm_q = xs[1] - part_q.unsqueeze(d)
        sm_p = log_sm_p.exp()
        return torch.stack(
            (part_p, part_q, torch.sum(xs[2].mul(sm_p) - log_sm_q.mul(sm_p), dim=d))
        )

    @staticmethod
    def mul(a, b):
        return torch.stack((a[0] + b[0], a[1] + b[1], a[2] + b[2]))

    @classmethod
    def prod(cls, xs, dim=-1):
        return xs.sum(dim)


class EntropySemiring(Semiring):
    """
    Implements an entropy expectation semiring.

    Computes both the log-values and the running distributional entropy.

    Based on descriptions in:

    * Parameter estimation for probabilistic finite-state transducers :cite:`eisner2002parameter`
    * First-and second-order expectation semirings with applications to minimum-risk training on translation forests :cite:`li2009first`
    * Sample Selection for Statistical Grammar Induction :cite:`hwa2000samplesf`
    """

    zero = torch.tensor([-1e5, 0.0])
    one = torch.tensor([0.0, 0.0])

    @staticmethod
    def size():
        return 2

    @staticmethod
    def convert(xs):
        values = torch.zeros((2,) + xs.shape).type_as(xs)
        values[0] = xs
        values[1] = 0
        return values

    @staticmethod
    def unconvert(xs):
        return xs[1]

    @staticmethod
    def sum(xs, dim=-1):
        assert dim != 0
        d = dim - 1 if dim > 0 else dim
        part = torch.logsumexp(xs[0], dim=d)
        log_sm = xs[0] - part.unsqueeze(d)
        sm = log_sm.exp()
        return torch.stack((part, torch.sum(xs[1].mul(sm) - log_sm.mul(sm), dim=d)))

    @staticmethod
    def mul(a, b):
        return torch.stack((a[0] + b[0], a[1] + b[1]))

    @classmethod
    def prod(cls, xs, dim=-1):
        return xs.sum(dim)


def TempMax(alpha):
    class _TempMax(_BaseLog):
        """
        Implements a max forward, hot softmax backward.
        """

        @staticmethod
        def sum(xs, dim=-1):
            pass

        @staticmethod
        def sparse_sum(xs, dim=-1):
            m, _ = torch.max(xs, dim=dim)
            a = torch.softmax(alpha * xs, dim)
            return m, (torch.zeros(a.shape[:-1]).long(), a)

    return _TempMax
