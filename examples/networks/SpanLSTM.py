import torch.nn as nn
import torch


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


class SpanLSTM(torch.nn.Module):
    """
    SpanLSTM model.
    """

    def __init__(self, NT, V, H):
        super().__init__()
        self.H = H
        self.V = V
        self.NT = NT
        self.emb = torch.nn.Embedding(V, H)
        self.lstm = torch.nn.LSTM(H, H, batch_first=True, bidirectional=True)
        self.res = Res(2 * H)
        self.proj = torch.nn.Linear(2 * H, NT)

    def forward(self, words, lengths):
        self.emb.weight[1].data.zero_()
        batch, N = words.shape

        f, ba = self.lstm(self.emb(words))[0].chunk(2, dim=2)
        a = torch.zeros(batch, N, N, self.H).type_as(f)
        b = torch.zeros(batch, N, N, self.H).type_as(f)
        a[:, :, : N - 1] = f[:, 1:].view(batch, 1, N - 1, self.H) - f[:].view(
            batch, N, 1, self.H
        )
        b[:, 1:, :] = ba[:, : N - 1].view(batch, N - 1, 1, self.H) - ba.view(
            batch, 1, N, self.H
        )
        out = self.proj(self.res(torch.cat([a, b], dim=-1)))
        return out
