import math
from tqdm import tqdm

import numpy as np
import torch

from fairseq.data import Dictionary

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')
print(len(dictionary))

bpe_cont = "@@"
bpe_toks = {
    i
    for i in range(len(dictionary))
    if dictionary[i].endswith(bpe_cont)
}

bpe_len = len(bpe_cont)

tokens = np.load('tokens.npy')
lm_scores = np.load('scores.npy')
knn_scores = np.load('knn_only_scores.npy')

assert len(tokens) == len(lm_scores)
assert len(knn_scores) == len(tokens)


# calculate unigram probability
# unigram_prob = np.load('unigram_prob.npy')
# unigram_scores = np.zeros(len(tokens))
#
# for idx in range(1, len(tokens)):
#     unigram_scores[idx] = np.log(unigram_prob[tokens[idx-1], tokens[idx]] + 1e-5)
#
# unigram_scores = unigram_scores.astype(np.float32)
# unigram_scores = torch.from_numpy(unigram_scores)

lm_scores = torch.from_numpy(lm_scores)
knn_scores = torch.from_numpy(knn_scores)

combine_probs = torch.stack([lm_scores, knn_scores], dim=0)

with open('small_interpolation_result.txt', 'w') as outfile:
    for lmbda in tqdm(np.linspace(0.0, 0.99, num=50)):
        coeffs = torch.ones_like(combine_probs)
        coeffs[0] = np.log(1 - lmbda)
        coeffs[1] = np.log(lmbda)

        scores = torch.logsumexp(combine_probs + coeffs, dim=0)

        score_sum = 0.
        count = 0

        tgt_len = tokens.size
        skipped_toks = 0
        for i in range(tgt_len - 1):
            if tokens[i].item() in bpe_toks:
                skipped_toks += 1
                scores[i + 1] += scores[i]
                scores[i] = 0

        score_sum += scores.sum()
        count += scores.numel() - skipped_toks

        avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
        outfile.write(str(lmbda) + '\t' + str(2**avg_nll_loss.item()) + '\n')

# combine_probs = torch.stack([lm_scores, kmeans_scores, unigram_scores], dim=0)
# with open('unigram_interpolation_result.txt', 'w') as outfile:
#     for lmbda1 in tqdm(np.linspace(0.01, 0.5, num=100)):
#         for lmbda2 in np.linspace(0.0001, 0.01, num=100):
#             coeffs = torch.ones_like(combine_probs)
#             coeffs[0] = np.log(1 - lmbda1 - lmbda2)
#             coeffs[1] = np.log(lmbda1)
#             coeffs[2] = np.log(lmbda2)
#
#             scores = torch.logsumexp(combine_probs + coeffs, dim=0)
#
#             score_sum = 0.
#             count = 0
#
#             tgt_len = tokens.size
#             skipped_toks = 0
#             for i in range(tgt_len - 1):
#                 if tokens[i].item() in bpe_toks:
#                     skipped_toks += 1
#                     scores[i + 1] += scores[i]
#                     scores[i] = 0
#
#             score_sum += scores.sum()
#             count += scores.numel() - skipped_toks
#
#             avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
#             outfile.write(str(lmbda1) + '\t' + str(lmbda2) + '\t' + str(2**avg_nll_loss.item()) + '\n')
