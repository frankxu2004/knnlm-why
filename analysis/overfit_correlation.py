import math

from scipy.stats import pearsonr, spearmanr
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

# calculate Pearson's correlation
corr, _ = pearsonr(lm_scores, knn_scores)
print('LM vs KNN Pearsons correlation: %.3f' % corr)

# calculate Pearson's correlation
corr, _ = spearmanr(lm_scores, knn_scores)
print('LM vs KNN Spearmans correlation: %.3f' % corr)


for ep in [10, 20, 30, 50, 100, 150, 200]:
    print(ep)
    overfit_scores = np.load('overfit_lm_scores_checkpoint' + str(ep) + '.npy')
    assert len(tokens) == len(lm_scores)
    assert len(knn_scores) == len(tokens)
    assert len(overfit_scores) == len(tokens)

    # calculate Pearson's correlation
    corr, _ = pearsonr(overfit_scores, knn_scores)
    print('OverfitLM vs KNN Pearsons correlation: %.3f' % corr)

    # calculate Pearson's correlation
    corr, _ = spearmanr(overfit_scores, knn_scores)
    print('OverfitLM vs KNN Spearmans correlation: %.3f' % corr)


