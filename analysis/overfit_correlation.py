import numpy as np
from scipy.stats import pearsonr, spearmanr

from fairseq.data import Dictionary
import torch

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
knn_scores = np.load('best_knn_only_scores.npy')
knnlm_scores = np.load('knnlm_scores.npy')

assert len(lm_scores) == len(tokens)
assert len(knn_scores) == len(tokens)

# calculate Pearson's correlation
corr, _ = pearsonr(lm_scores, knn_scores)
print('LM vs KNN Pearsons correlation: %.3f' % corr)

# calculate Pearson's correlation
corr, _ = spearmanr(lm_scores, knn_scores)
print('LM vs KNN Spearmans correlation: %.3f' % corr)


for ep in [2, 5, 10, 15, 18]:
    print('Epoch', ep)
    overfit_scores = np.load('overfit_lm_scores_checkpoint' + str(ep) + '.npy')
    assert len(tokens) == len(lm_scores)
    assert len(knn_scores) == len(tokens)
    assert len(overfit_scores) == len(tokens)

    overfit_probs = np.exp(overfit_scores)
    lm_probs = np.exp(lm_scores)
    knn_probs = np.exp(knn_scores)

    knn_lm_diff = knn_probs - lm_probs
    overfit_lm_diff = overfit_probs - lm_probs

    # calculate Pearson's correlation
    corr, _ = pearsonr(overfit_lm_diff, knn_lm_diff)
    print('OverfitLM vs KNN Pearsons correlation: %.3f' % corr)

    # calculate Pearson's correlation
    corr, _ = spearmanr(overfit_lm_diff, knn_lm_diff)
    print('OverfitLM vs KNN Spearmans correlation: %.3f' % corr)


