import torch
from .semirings import MultiSampledSemiring, MaxSemiring


class SelfCritical:
    def __init__(self, struct, reward_fn):
        self.struct = struct
        self.reward_fn = reward_fn
        self.max_fn = self.struct(MaxSemiring)
        self.sample_fn = self.struct(MultiSampledSemiring)

    def forward(self, phi, lengths, K=5):
        sample = self.sample_fn.marginals(phi, lengths=lengths)
        sample = sample.detach()
        trees = []
        samples = []
        for k in range(K):
            tmp_sample = MultiSampledSemiring.to_discrete(sample, k+1)
            samples.append(tmp_sample)
            sampled_tree = self.max_fn.from_parts(tmp_sample)[0].cpu()
            trees.append(sampled_tree)
        structs = torch.stack(samples)
        argmax = self.max_fn.marginals(phi, lengths=lengths)    
        argmax_tree = self.max_fn.from_parts(argmax.detach())[0].cpu()
        trees.append(argmax_tree)
        sample_score = self.reward_fn(torch.cat(trees), K+1)
        total = sample_score[:-1].mean(dim=0)
        max_score = sample_score[-1].clone().detach()
        rewards = sample_score[:-1] - max_score.view(1, sample_score.shape[1])
        
        return structs, rewards.detach(), total, max_score
