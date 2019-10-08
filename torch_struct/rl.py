import torch


class SelfCritical:
    def __init__(self, reward_fn):
        self.reward_fn = reward_fn

    def forward(self, dist, K=5):
        samples = dist.sample((K,))
        trees = []
        for k in range(K):
            sampled_tree = dist.struct.from_parts(samples[k])[0].cpu()
            trees.append(sampled_tree)
        structs = torch.stack(trees)
        argmax = dist.argmax
        argmax_tree = dist.struct.from_parts(argmax.detach())[0].cpu()
        trees.append(argmax_tree)
        sample_score = self.reward_fn(torch.cat(trees), K + 1)
        total = sample_score[:-1].mean(dim=0)
        max_score = sample_score[-1].clone().detach()
        rewards = sample_score[:-1] - max_score.view(1, sample_score.shape[1])

        return structs, rewards.detach(), total, max_score
