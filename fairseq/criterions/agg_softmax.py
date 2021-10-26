# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F
import torch

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion


@register_criterion('agg_softmax')
class AggSoftmaxCriterion(CrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        num_special_tokens = task.target_dictionary.nspecial
        coef = torch.zeros((num_special_tokens + args.pseudo_vocab_ratio *
                            (len(task.target_dictionary) - num_special_tokens), len(task.target_dictionary)),
                           dtype=torch.bool)
        for i in range(coef.shape[1]):
            if i < num_special_tokens:
                coef[i, i] = 1
            else:
                coef[range()]
        print(coef.shape)
        exit()

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=False)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        print(lprobs.shape)
        target = model.get_targets(sample, net_output).view(-1)
        print(target.shape)
        print(self.padding_idx)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss
