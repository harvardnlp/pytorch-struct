from torch_struct import TreeCRF, SelfCritical
import torchtext.data as data
from torch_struct.data import ListOpsDataset, TokenBucket
from torch_struct.networks import TreeLSTM, SpanLSTM
import torch
import torch.nn as nn
import argparse
from argparse import ArgumentParser
import random

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

PRIOR = 0
COND = 1
class ListOpsModel(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.WORD, self.LABEL, self.TREE = None, None, None
        self.hparams = hparams
        print(self.hparams)
        self.train_dataloader()
        NT = 1
        T = len(self.WORD.vocab)
        V = T
        self.prior = SpanLSTM(NT, len(self.WORD.vocab), self.hparams.hidden_size)
        self.conditional = TreeLSTM(
            self.hparams.hidden_size, len(self.WORD.vocab) + 100, len(self.LABEL.vocab)
        )
        self.prior_dist = TreeCRF
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self):
        pass

    def configure_optimizers(self):
        opt_prior = torch.optim.Adadelta(
            list(self.prior.parameters()),
            lr=self.hparams.prior_lr)
        opt_conditional = torch.optim.Adadelta(
            list(self.conditional.parameters()),
            lr=self.hparams.conditional_lr
        )
        return [opt_prior, opt_conditional], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        print(batch)
        print(batch.word)
        print(batch.label)
        words, lengths = batch.word
        label = batch.label
        batch_size = label.shape[0]
        _, N = words.shape

        def tree_reward(spans, K):
            new_spans = expand_spans(spans, words, K, V)
            g, labels, indices, topo = TreeLSTM.spans_to_dgl(new_spans)
            ret = tree_lstm(
                g, labels, indices, topo, torch.cat([lengths for _ in range(K)])
            )
            ret = ret.view(K, batch_size, -1)
            return -ret[:, torch.arange(batch_size), label].view(K, batch_size)

        if optimizer_idx == PRIOR:
            sc = SelfCritical(tree_reward)
            phi = self.model(words, lengths)
            dist = self.prior_dist(phi)
            structs, rewards, score, max_score = sc.forward(dist, K=self.hparams.rl_samples)
            loss = score.mean()
            print(loss)
            return dict(loss= loss)


        elif optimizer_idx == COND:
            entropy = dist.entropy
            r = dist.log_prob(structs)
            obj = rewards.mul(r).mean(-1).mean(-1)
            policy = (
                obj - self.hparams.entropy * entropy.div(lengths.float().cuda()).mean()
            )
            print(policy)
            return dict(loss= policy)


    def val_step(self, batch):
        words, lengths = batch.word
        trees = batch.tree
        label = batch.label
        words = words.cuda()

        def tree_reward(spans):
            new_spans = expand_spans(spans, words, 1, V)
            g, labels, indices, topo = TreeLSTM.spans_to_dgl(new_spans)
            _, am = tree_lstm(g, labels, indices, topo, lengths).max(-1)
            return (label == am).sum(), label.shape[0]

        words = words.cuda()
        phi = model(words, lengths)
        dist = TreeCRF(phi, lengths)
        argmax = dist.argmax
        argmax_tree = dist.struct.from_parts(argmax.detach())[0]
        score, tota = tree_reward(argmax_tree)
        total += int(tota)
        correct += score

    def train_dataloader(self):
        self.WORD = data.Field(
            include_lengths=True, batch_first=True, eos_token=None, init_token=None
        )
        self.LABEL = data.Field(sequential=False, batch_first=True)
        TREE = data.RawField(postprocessing=ListOpsDataset.tree_field(self.WORD))
        TREE.is_target = False
        train = ListOpsDataset(
            "data/train_temp.tsv",
            (("word", self.WORD), ("label", self.LABEL), ("tree", self.TREE)),
            filter_pred=lambda x: 5 < len(x.word) < self.hparams.max_train_len
        )
        self.WORD.build_vocab(train)
        self.LABEL.build_vocab(train)
        train_iter = TokenBucket(
            train, batch_size=1500, device="cpu",#device="cuda:0",
            key=lambda x: len(x.word)
        )
        all_data = list([t for t in train_iter])
        print("all data")
        train_loader = torch.utils.data.DataLoader(
            dataset=all_data,
            batch_size=None,
            batch_sampler=None,
            shuffle=None,
            num_workers=0,
            sampler=None
        )
        return train_loader

    def val_dataloader(self):
        print("val")
        valid = ListOpsDataset(
            "data/test_temp.tsv",
            (("word", self.WORD), ("label", self.LABEL), ("tree", self.TREE)),
            filter_pred=lambda x: 5 < len(x.word) < 150,
        )
        valid_iter = data.BucketIterator(
            valid, batch_size=50, train=False, sort=False, device="cpu" #device="cuda:0"
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=[t for t in valid_iter],
            batch_size=None,
            shuffle=None,
            num_workers=0,
            batch_sampler=None
        )
        return valid_loader

    def test_dataloader(self):
        return self.val_dataloader()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=42,
                            help='seed for initializing training. ')
        parser.add_argument('--method', type=str, default="reinforce",
                            help='training method. ')
        parser.add_argument('--prior_lr', type=float, default=0.1,
                            help='lr for structured prior. ')
        parser.add_argument('--conditional_lr', type=float, default=1.0,
                            help='lr for conditional model. ')
        parser.add_argument('--entropy_regularizer', type=float, default=0.001,
                            help='entropy regularizer weight. ')
        parser.add_argument('--rl_samples', type=int, default=5,
                            help='samples of reinforce to take. ')
        parser.add_argument('--hidden_size', type=int, default=100,
                            help='model hidden size')
        parser.add_argument('--max_train_len', type=int, default=100,
                            help='maximum training length')


        return parser

def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--data-path', metavar='DIR', default="data/", type=str,
                               help='path to dataset')
    parent_parser.add_argument('--save-path', metavar='DIR', default=".", type=str,
                               help='path to save output')
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='how many gpus')

    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parser = ListOpsModel.add_model_specific_args(parent_parser)
    return parser.parse_args()

def main(hparams):
    model = ListOpsModel(hparams)
    if hparams.seed is not None:
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        # cudnn.deterministic = True
    trainer = pl.Trainer(
        default_save_path=hparams.save_path,
        gpus=hparams.gpus,
        # val_check_interval=1.0,
        max_epochs=hparams.epochs,
        # fast_dev_run=True
    )
    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)


if __name__ == '__main__':
    main(get_args())
