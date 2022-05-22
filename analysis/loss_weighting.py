import json

import numpy as np

from fairseq.data import Dictionary

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')
print(len(dictionary))


tokens = np.load('train_tokens.npy').astype(np.int32)

token_freqs = [0] * len(dictionary)
token_knn_helped = [0] * len(dictionary)

for tid in tokens:
    token_freqs[tid] += 1

print('tok freq count done')


lm_scores = np.load('train_scores.npy')
knn_scores = np.load('train_knn_scores.npy')

for t, s, ks in zip(tokens, lm_scores, knn_scores):
    if ks > s:
        token_knn_helped[t] += 1

for i in range(len(dictionary)):
    if token_freqs[i] == 0:
        token_knn_helped[i] = 0
    else:
        token_knn_helped[i] = token_knn_helped[i] / token_freqs[i]

with open('analysis/train_knn_helped.json', 'w', encoding='utf-8') as outfile:
    json.dump(token_knn_helped, outfile)

