import torch
import torch.nn as nn


class Res(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.u1 = nn.Linear(H, H)
        self.u2 = nn.Linear(H, H)

        self.v1 = nn.Linear(H, H)
        self.v2 = nn.Linear(H, H)
        self.w = nn.Linear(H, H)

    def forward(self, y):
        y = self.w(y)
        y = y + torch.relu(self.v1(torch.relu(self.u1(y))))
        return y + torch.relu(self.v2(torch.relu(self.u2(y))))


class NeuralCFG(torch.nn.Module):
    """
    NeuralCFG From Kim et al
    """

    def __init__(self, V, T, NT, H):
        super().__init__()
        self.NT = NT
        self.V = V
        self.T = T
        self.word_emb = nn.Parameter(torch.Tensor(V, H))
        self.term_emb = nn.Parameter(torch.Tensor(T, H))
        self.nonterm_emb = nn.Parameter(torch.Tensor(NT, H))
        self.nonterm_emb_c = nn.Parameter(torch.Tensor(NT + T, NT + T, H))
        self.root_emb = nn.Parameter(torch.Tensor(NT, H))
        self.s_emb = nn.Parameter(torch.Tensor(1, H))
        self.mlp1 = Res(H)
        self.mlp2 = Res(H)
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, input):
        T, NT = self.T, self.NT

        def terms(words):
            b, n = input.shape[:2]
            term_prob = (
                torch.einsum("vh,th->tv", self.word_emb, self.mlp1(self.term_emb))
                .log_softmax(-1)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(b, n, self.T, self.V)
            )
            indices = input.unsqueeze(2).expand(b, n, self.T).unsqueeze(3)
            term_prob = torch.gather(term_prob, 3, indices).squeeze(3)
            return term_prob

        def rules(b):
            return (
                torch.einsum("sh,tuh->stu", self.nonterm_emb, self.nonterm_emb_c)
                .view(NT, -1)
                .log_softmax(-1)
                .view(1, NT, NT + T, NT + T)
                .expand(b, NT, NT + T, NT + T)
            )

        def roots(b):
            return (
                torch.einsum("ah,th->t", self.s_emb, self.mlp2(self.root_emb))
                .log_softmax(-1)
                .view(1, NT)
                .expand(b, NT)
            )

        batch = input.shape[0]
        return terms(input), rules(batch), roots(batch)
