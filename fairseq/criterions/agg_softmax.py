# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F
import torch
import numpy as np
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
        self.coef = None
        if self.ratio > 1:
            print('Using one hot cluster distribution with K=', args.pseudo_vocab_ratio)
            # one-hot coef
            self.coef = self.initialize_projection_matrix(task.target_dictionary, args.pseudo_vocab_ratio)
            self.coef = self.coef.to_dense().bool()
            if torch.cuda.is_available() and not args.cpu:
                self.coef = self.coef.cuda()
        if args.load_centroid_distribution:
            # load prior coef
            from scipy import sparse
            print('Loading cluster-token distribution from file:', args.load_centroid_distribution)
            freq_mat = sparse.load_npz(args.load_centroid_distribution).tocoo().T
            values = freq_mat.data
            indices = np.vstack((freq_mat.row, freq_mat.col))
            self.coef = torch.sparse_coo_tensor(indices, values.astype(np.float32),
                                                freq_mat.shape).coalesce()
            if torch.cuda.is_available() and not args.cpu:
                self.coef = self.coef.cuda().to_dense()

        print('coef is:')
        print(self.coef)

    @staticmethod
    def initialize_projection_matrix(dictionary, ratio):
        num_special_tokens = dictionary.nspecial
        vocab_size = len(dictionary)

        num_out_emb_entries = num_special_tokens + ratio * (
                vocab_size - num_special_tokens)

        indexes = []
        values = []
        for i in range(vocab_size):
            if i < num_special_tokens:
                indexes.append((i, i))
                values.append(1.)
            else:
                for j in range(num_special_tokens + ratio * (i - num_special_tokens),
                               num_special_tokens + ratio * (i - num_special_tokens + 1)):
                    indexes.append((i, j))
                    values.append(1.)
        return torch.sparse_coo_tensor(list(zip(*indexes)), values,
                                       (vocab_size, num_out_emb_entries)).coalesce()

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output).view(-1)
        if self.coef is None:
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
        else:
            lprobs = model.get_normalized_probs(net_output, log_probs=False)
            lprobs = lprobs.view(-1, lprobs.size(-1))  # bsz x clusters
            print(target.shape)
            lprobs = torch.log(torch.clamp((self.coef[target] * lprobs).sum(-1), min=1e-9))  # bsz x vocab
        loss = - lprobs.sum()

        # loss = F.nll_loss(
        #     lprobs,
        #     target,
        #     ignore_index=self.padding_idx,
        #     reduction='sum' if reduce else 'none',
        # )
        return loss, loss
