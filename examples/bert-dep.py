#!/usr/bin/env python3

import torchtext
import torch

from torch import nn
from torch_struct import DependencyCRF
from torch_struct.data import SubTokenizedField, ConllXDataset, TokenBucket
from torchtext.data import RawField, BucketIterator

from pytorch_transformers import BertModel, BertTokenizer, AdamW, WarmupLinearSchedule

config = {'bert': 'bert-base-cased', 'H': 768, 'dropout': 0.2}

# parse conll dependency data
model_class, tokenizer_class, pretrained_weights = BertModel, BertTokenizer, config['bert']
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

def batch_num(nums):
    lengths = torch.tensor([len(n) for n in nums]).long()
    n = lengths.max()
    out = torch.zeros(len(nums), n).long()
    for b, n in enumerate(nums):
        out[b, :len(n)] = torch.tensor(n)
    return out, lengths

HEAD = RawField(preprocessing=lambda x: [int(i) for i in x],
        postprocessing=batch_num)
HEAD.is_target = True
WORD = SubTokenizedField(tokenizer)

def len_filt(x): return 5 < len(x.word[0]) < 40

train = ConllXDataset('wsj.train.conllx', (('word', WORD), ('head', HEAD)),
        filter_pred=len_filt)
train_iter = TokenBucket(train, 750)
val = ConllXDataset('wsj.dev.conllx', (('word', WORD), ('head', HEAD)),
        filter_pred=len_filt)
val_iter = BucketIterator(val, batch_size=20, device='cuda:0')

# make bert model to compute potentials
H = config['H']
class Model(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.base_model = model_class.from_pretrained(pretrained_weights)
        self.linear = nn.Linear(H, H)
        self.bilinear = nn.Linear(H, H)
        self.root = nn.Parameter(torch.rand(H))
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, words, mapper):
        out = self.dropout(self.base_model(words)[0])
        out = torch.matmul(mapper.float().cuda().transpose(1, 2), out)
        final1 = torch.matmul(out, self.linear.weight)
        final2 = torch.einsum('bnh,hg,bmg->bnm', out, self.bilinear.weight, final1)
        root_score = torch.matmul(out, self.root)
        final2 = final2[:, 1:-1, 1:-1]
        N = final2.shape[1]
        final2[:, torch.arange(N), torch.arange(N)] += root_score[:, 1:-1]
        return final2

model = Model(H)
model.cuda()

# validation and train loops
def validate(val_iter):
    incorrect_edges = 0
    total_edges = 0
    model.eval()
    for i, ex in enumerate(val_iter):
        words, mapper, _ = ex.word
        label, lengths = ex.head
        batch, _ = label.shape

        final = model(words.cuda(), mapper)
        for b in range(batch):
            final[b, lengths[b]-1:, :] = 0
            final[b, :, lengths[b]-1:] = 0
        dist = DependencyCRF(final, lengths=lengths)
        gold = dist.struct.to_parts(label, lengths=lengths).type_as(dist.argmax)
        incorrect_edges += (dist.argmax[:, :].cpu() - gold[:, :].cpu()).abs().sum() / 2.0
        total_edges += gold.sum()

    print(total_edges, incorrect_edges)
    model.train()

def train(train_iter, val_iter, model):
    opt = AdamW(model.parameters(), lr=1e-4, eps=1e-8)
    scheduler = WarmupLinearSchedule(opt, warmup_steps=20, t_total=2500)
    model.train()
    losses = []
    for i, ex in enumerate(train_iter):
        opt.zero_grad()
        words, mapper, _ = ex.word
        label, lengths = ex.head
        batch, _ = label.shape

        # Model
        final = model(words.cuda(), mapper)
        for b in range(batch):
            final[b, lengths[b]-1:, :] = 0
            final[b, :, lengths[b]-1:] = 0

        if not lengths.max() <= final.shape[1] + 1:
            print("fail")
            continue
        dist = DependencyCRF(final, lengths=lengths)

        labels = dist.struct.to_parts(label, lengths=lengths).type_as(final)
        log_prob = dist.log_prob(labels)

        loss = log_prob.sum()
        (-loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()
        scheduler.step()
        losses.append(loss.detach())
        if i % 50 == 1:
            print(-torch.tensor(losses).mean(), words.shape)
            losses = []
        if i % 600 == 500:
            validate(val_iter)

train(train_iter, val_iter, model)
