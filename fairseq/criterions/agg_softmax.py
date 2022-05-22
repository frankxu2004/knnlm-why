# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import json
import torch.nn.functional as F

from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion


@register_criterion('agg_softmax')
class AggSoftmaxCriterion(CrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.num_special_tokens = task.target_dictionary.nspecial
        self.vocab_size = len(task.target_dictionary)
        self.coef = None
        self.args = args
        if args.pseudo_vocab_ratio > 1:
            print('Using one hot cluster distribution with K=', args.pseudo_vocab_ratio)
            # one-hot coef
            self.coef = self.initialize_projection_matrix(task.target_dictionary, args.pseudo_vocab_ratio)
            self.coef = self.coef.to_dense().bool()  # save memory using bool for one hot
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
                                                freq_mat.shape).coalesce().to_dense()
            if torch.cuda.is_available() and not args.cpu:
                self.coef = self.coef.cuda()

        if args.num_extra_embed_file:
            print('Loading number of extra embeddings per word from file:', args.num_extra_embed_file)
            self.coef = self.initialize_projection_matrix(task.target_dictionary, args.pseudo_vocab_ratio,
                                                          num_extra_embed_file=args.num_extra_embed_file)
            self.coef = self.coef.to_dense().bool()  # save memory using bool for one hot
            if torch.cuda.is_available() and not args.cpu:
                self.coef = self.coef.cuda()

        print('coef is:')
        print(self.coef)
        if self.coef is not None:
            print('coef shape:')
            print(self.coef.shape)

        self.lmbda = 0.25

    @staticmethod
    def initialize_projection_matrix(dictionary, ratio, num_extra_embed_file=None):
        vocab_size = len(dictionary)

        if num_extra_embed_file:
            num_extra_embeds = json.load(open(num_extra_embed_file))
            assert len(num_extra_embeds) == vocab_size
            num_out_emb_entries = vocab_size + sum(num_extra_embeds)
            indexes = []
            values = []
            for i in range(vocab_size):
                indexes.append((i, i))
                values.append(1.)
            added_extra = 0
            for i in range(vocab_size):
                for j in range(num_extra_embeds[i]):
                    indexes.append((i, vocab_size+added_extra))
                    values.append(1.)
                    added_extra += 1

        else:
            num_special_tokens = dictionary.nspecial

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
                                       (vocab_size, num_out_emb_entries), dtype=torch.int8).coalesce()

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output).view(-1)
        if self.args.interpolated_loss:
            x, extra = net_output
            assert len(x) == 2
            net_output = x[0], extra
            net_output_orig = x[1], extra
            lprobs_orig = model.get_normalized_probs(net_output_orig, log_probs=True)
            lprobs_orig = lprobs_orig.view(-1, lprobs_orig.size(-1))

        if self.coef is None:
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))

            if self.args.interpolated_loss:
                combine_probs = torch.stack([lprobs_orig, lprobs], dim=0)
                coeffs = torch.ones_like(combine_probs)
                coeffs[0] = np.log(1 - self.lmbda)
                coeffs[1] = np.log(self.lmbda)
                lprobs = torch.logsumexp(combine_probs + coeffs, dim=0)

            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction='sum' if reduce else 'none',
            )

        else:
            lprobs = model.get_normalized_probs(net_output, log_probs=False)
            lprobs = lprobs.view(-1, lprobs.size(-1))  # bsz x nV
            lprobs = torch.log(torch.clamp((self.coef[target] * lprobs).sum(-1), min=1e-9))  # bsz x vocab

            lprobs_orig = lprobs_orig[range(lprobs_orig.size(0)), target]

            if self.args.interpolated_loss:
                combine_probs = torch.stack([lprobs_orig, lprobs], dim=0)
                coeffs = torch.ones_like(combine_probs)
                coeffs[0] = np.log(1 - self.lmbda)
                coeffs[1] = np.log(self.lmbda)
                lprobs = torch.logsumexp(combine_probs + coeffs, dim=0)

            loss = - lprobs.sum()

        return loss, loss
