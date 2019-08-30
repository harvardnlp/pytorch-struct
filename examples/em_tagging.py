import torch
import torch_struct

def fit(params, data):
    "Run EM / forward to fit the params to the data."
    for p in params.values():
        p.requires_grad_(True)
    opt = torch.optim.Adam(params.values(), lr=0.01)

    for epoch in range(1000):
        opt.zero_grad()
        scores = torch_struct.hmm(params["trans"].softmax(dim=0),
                                  params["emit"].softmax(dim=0),
                                  params["init"].softmax(dim=0),
                                  data)
        loss = -torch_struct.linearchain_inside(scores.log())[0].mean()
        loss.backward()
        opt.step()
        if epoch % 100:
            print(-loss.detach())


            # Compute sample
            z = torch_struct.linearchain(scores.log(),
                                         semiring=torch_struct.SampledSemiring)
            print(z.nonzero()[:, 2])

# Create the data for the experiments.

def random_parameters(V, C):
    transition = torch.rand(C, C)
    emission = torch.rand(V, C)
    init = torch.rand(C)
    return {"trans": transition,
            "emit": emission,
            "init": init}

def easy_parameters(V, C):
    transition = (torch.eye(C, C) + 1e-5).log()
    emission = (torch.eye(C, C) + 1e-5).log()
    init = torch.ones(C)
    return {"trans": transition,
            "emit": emission,
            "init": init}


def sample(logits):
    return torch.distributions.Categorical(logits=logits).sample()

def generate_data(params, N, total):
    xs = []
    zs = []
    for t in range(total):
        z_n = sample(params["init"])
        x = []
        z = [z_n]
        for n in range(N):
            x.append(sample(params["emit"][z[-1]]))
            z.append(sample(params["trans"][z[-1]]))
        xs.append(x), zs.append(z)
    return torch.tensor(xs), torch.tensor(zs)


def main():
    V, C, N, total = 5, 5, 10, 100
    true_params = easy_parameters(V, C)
    x, z = generate_data(true_params, N, total)
    params = random_parameters(V, C)
    fit(params, x)
main()
