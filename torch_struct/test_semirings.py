# smint = integers(min_value=2, max_value=20)
# @given(smint, smint, smint)
# def test_mat(a, b, c):
#     m1 = torch.rand(a, b)
#     m2 = torch.rand(b, c)
#     assert(torch.isclose(StdSemiring.contract("ab,bc->ac", m1, m2),
#             torch.einsum("ab,bc->ac", (m1, m2))).all())
#     assert(torch.isclose(StdSemiring.contract("ab,bc->a", m1, m2),
#                          torch.einsum("ab,bc->a", (m1, m2))).all())
#     assert(torch.isclose(StdSemiring.contract("ab,bc->c", m1, m2),
#             torch.einsum("ab,bc->c", (m1, m2))).all())
#     assert(torch.isclose(StdSemiring.contract("ab,bc->", m1, m2),
#             torch.einsum("ab,bc->", (m1, m2))).all())

# @given(smint, smint, smint, smint, smint, smint)
# def test_tensor(a, b, c, d, e, f):
#     m1 = torch.rand(a, b, c)

#     m2 = torch.rand(d, e, f)

#     assert(StdSemiring.contract("abc,def->ac", m1, m2).shape ==
#            torch.einsum("abc,def->ac", (m1, m2)).shape)

#     assert(torch.isclose(StdSemiring.contract("abc,def->ac", m1, m2),
#             torch.einsum("abc,def->ac", (m1, m2))).all())

#     m2 = torch.rand(a, b)
#     print(StdSemiring.contract("abc,ab->ac", m1, m2),
#                          torch.einsum("abc,ab->ac", (m1, m2)))
#     assert(torch.isclose(StdSemiring.contract("abc,ab->ac", m1, m2),
#                          torch.einsum("abc,ab->ac", (m1, m2))).all())
