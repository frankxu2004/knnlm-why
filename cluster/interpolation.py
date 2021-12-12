import math

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
scores = np.load('scores.npy')
kmeans_tokens = np.load('kmeans_tokens.npy')
kmeans_scores = np.load('kmeans_scores.npy')

assert len(tokens) == len(scores)
assert len(kmeans_tokens) == len(tokens)


scores = torch.from_numpy(scores)
kmeans_scores = torch.from_numpy(kmeans_scores)

with open('interpolation_result.txt', 'w') as outfile:
    for lmbda in np.linspace(0.0, 1.0, num=50):

        combine_probs = torch.stack([scores, kmeans_scores], dim=0)
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
