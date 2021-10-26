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
        num_special_tokens = task.target_dictionary.nspecial

        num_out_emb_entries = num_special_tokens + args.pseudo_vocab_ratio * (
                    len(task.target_dictionary) - num_special_tokens)

        indexes = []
        values = []
        for i in range(len(task.target_dictionary)):
            if i < num_special_tokens:
                indexes.append((i, i))
                values.append(1.)
            else:
                for j in range(num_special_tokens + args.pseudo_vocab_ratio * (i - num_special_tokens),
                               num_special_tokens + args.pseudo_vocab_ratio * (i - num_special_tokens + 1)):
                    indexes.append((i, j))
                    values.append(1.)
        self.coef = torch.sparse_coo_tensor(list(zip(*indexes)), values,
                                            (len(task.target_dictionary), num_out_emb_entries))
        if torch.cuda.is_available() and not args.cpu:
            self.coef = self.coef.cuda()

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        lprobs = torch.log(torch.sparse.mm(self.coef, lprobs.T).T)
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss
