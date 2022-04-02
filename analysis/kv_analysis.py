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

assert len(tokens) == len(lm_scores)

lm_scores = torch.from_numpy(lm_scores).cuda()
tgt_len = tokens.size
skipped_toks = 0
for i in range(tgt_len - 1):
    if tokens[i].item() in bpe_toks:
        skipped_toks += 1

count = len(tokens) - skipped_toks

knn_helping = 0
with open('kv_interpolation.txt', 'w') as outfile:
    for f in ['best_knn_only_scores.npy', 'kv_scores/2v_scores.npy',
              'kv_scores/3v_scores.npy',
              'additional_linear_scores/add_linear_scores.npy',
              'additional_linear_scores/additional_softmax_scores.npy']:
        overfit_scores = np.load(f)
        overfit_scores = torch.from_numpy(overfit_scores).cuda()
        combine_probs = torch.stack([lm_scores, overfit_scores], dim=0)

        oracle_scores, argmaxs = torch.max(combine_probs, dim=0)

        oracle_ppl = torch.exp(-oracle_scores.sum() / count)

        if 'knn' in f:
            knn_helping = argmaxs

        match_knn = torch.sum(argmaxs == knn_helping).item() / len(tokens)
        extra_helping_percentage = torch.sum(argmaxs).item() / len(tokens)

        knn_helping_scores = -(combine_probs[0][knn_helping == 0].sum() +
                               combine_probs[1][knn_helping == 1].sum())

        knn_helping_ppl = torch.exp(knn_helping_scores / count)

        best_ppl = 1e10
        best_lmbda = 0
        for lmbda in np.linspace(0.0, 0.999, num=200):
            coeffs = torch.ones_like(combine_probs)
            coeffs[0] = np.log(1 - lmbda)
            coeffs[1] = np.log(lmbda)

            scores = torch.logsumexp(combine_probs + coeffs, dim=0)

            score_sum = scores.sum()

            avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
            ppl = 2 ** avg_nll_loss.item()
            if ppl < best_ppl:
                best_ppl = ppl
                best_lmbda = lmbda

        outfile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(f, best_lmbda, best_ppl, oracle_ppl,
                                                            match_knn, extra_helping_percentage,
                                                            knn_helping_ppl))
