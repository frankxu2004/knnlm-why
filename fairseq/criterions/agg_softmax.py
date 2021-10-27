# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F
import torch
from torch import nn

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion


@register_criterion('agg_softmax')
class AggSoftmaxCriterion(CrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.num_special_tokens = task.target_dictionary.nspecial
        self.vocab_size = len(task.target_dictionary)
        self.ratio = args.pseudo_vocab_ratio
        num_out_emb_entries = self.num_special_tokens + args.pseudo_vocab_ratio * (
                    self.vocab_size - self.num_special_tokens)

        indexes = []
        values = []
        for i in range(len(task.target_dictionary)):
            if i < self.num_special_tokens:
                indexes.append((i, i))
                values.append(1.)
            else:
                for j in range(self.num_special_tokens + self.ratio * (i - self.num_special_tokens),
                               self.num_special_tokens + self.ratio * (i - self.num_special_tokens + 1)):
                    indexes.append((i, j))
                    values.append(1.)
        self.coef = torch.sparse_coo_tensor(list(zip(*indexes)), values,
                                            (self.vocab_size, num_out_emb_entries))
        if torch.cuda.is_available() and not args.cpu:
            self.coef = self.coef.cuda()

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        lprobs = torch.log(torch.clamp(torch.sparse.mm(self.coef, lprobs.T).T, min=1e-9)) #bsz x vocab
        target = model.get_targets(sample, net_output).view(-1)
        # agg_probs = torch.full((target.size(0), self.ratio), -10000).cuda()
        # for idx, t in enumerate(target):
        #     if t < self.num_special_tokens:
        #         agg_probs[idx, 0] = lprobs[idx, t]
        #     else:
        #         agg_probs[idx, range(self.ratio)] = lprobs[idx, range(self.num_special_tokens + self.ratio * (t - self.num_special_tokens),
        #                        self.num_special_tokens + self.ratio * (t - self.num_special_tokens + 1))]
        # agg_probs = torch.logsumexp(agg_probs, dim=-1)
        # if reduce:
        #     loss = torch.sum(-agg_probs[target != self.padding_idx])
        # else:
        #     loss = -agg_probs[target != self.padding_idx]

        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss
