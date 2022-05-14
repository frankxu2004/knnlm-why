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

tgt_len = tokens.size
skipped_toks = 0
for i in range(tgt_len - 1):
    if tokens[i].item() in bpe_toks:
        skipped_toks += 1

count = len(tokens) - skipped_toks

lm_scores = np.load('scores.npy')
another_lm_scores = np.load('seed3_scores.npy')
third_lm_scores = np.load('kv_scores/2v_scores.npy')
knn_scores = np.load('best_knn_only_scores.npy')

assert len(tokens) == len(lm_scores)
assert len(knn_scores) == len(tokens)
assert len(another_lm_scores) == len(tokens)


lm_scores = torch.from_numpy(lm_scores).cuda()
another_lm_scores = torch.from_numpy(another_lm_scores).cuda()
third_lm_scores = torch.from_numpy(third_lm_scores).cuda()
knn_scores = torch.from_numpy(knn_scores).cuda()


combine_probs = torch.stack([lm_scores, another_lm_scores, third_lm_scores, knn_scores], dim=0)
best_ppl = 1e10
best_lmbda1 = 0
best_lmbda2 = 0
best_lmbda3 = 0

with open('multi_way_interpolation_result.txt', 'w') as outfile:
    for lmbda1 in tqdm(np.linspace(0.01, 0.99, num=100)):
        for lmbda2 in np.linspace(0.01, 1 - lmbda1, num=100):
            for lmbda3 in np.linspace(0.01, 1 - lmbda1 - lmbda2, num=100):

                coeffs = torch.ones_like(combine_probs)
                coeffs[0] = np.log(1 - lmbda1 - lmbda2 - lmbda3)
                coeffs[1] = np.log(lmbda1)
                coeffs[2] = np.log(lmbda2)
                coeffs[3] = np.log(lmbda3)

                scores = torch.logsumexp(combine_probs + coeffs, dim=0)

                score_sum = scores.sum()

                avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
                ppl = 2 ** avg_nll_loss.item()
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_lmbda1 = lmbda1
                    best_lmbda2 = lmbda2
                    best_lmbda3 = lmbda3

                outfile.write(str(lmbda1) + '\t' + str(lmbda2) + '\t' + str(2**avg_nll_loss.item()) + '\n')

print(best_ppl, best_lmbda1, best_lmbda2, best_lmbda3)
