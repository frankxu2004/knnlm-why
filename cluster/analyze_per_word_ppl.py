import math

import numpy as np

from fairseq.data import Dictionary

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')
print(len(dictionary))

tokens = np.load('tokens.npy')
scores = np.load('scores.npy')

assert len(tokens) == len(scores)

bins = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
x = ['10', '100', '1k', '10k', '100k', '1M', '1M+']
avg_scores = [[] for _ in range(len(x))]
for wordid, freq in enumerate(dictionary.count):
    s = scores[tokens == wordid]
    if len(s) > 0:
        avg_score = - np.mean(scores[tokens==wordid]) / math.log(2)
        bin_id = len(bins)
        for idx in range(len(bins)):
            if freq <= bins[idx]:
                bin_id = idx
                break
        avg_scores[bin_id].append(avg_score)

for bin_id, scores in enumerate(avg_scores):
    print(x[bin_id], np.mean(scores))
