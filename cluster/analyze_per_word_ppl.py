import math

import numpy as np

from fairseq.data import Dictionary

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')
print(len(dictionary))

tokens = np.load('tokens.npy')
scores = np.load('scores.npy')
kmeans_tokens = np.load('kmeans_tokens.npy')
kmeans_scores = np.load('kmeans_scores.npy')

assert len(tokens) == len(scores)

bins = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
x = ['10', '100', '1k', '10k', '100k', '1M', '1M+']


def calc_mean_loss(tokens, scores):
    means = []
    avg_scores = [[] for _ in range(len(x))]
    for wordid, freq in enumerate(dictionary.count):
        s = scores[tokens == wordid]
        if len(s) > 0:
            avg_score = - np.mean(scores[tokens == wordid]) / math.log(2)
            bin_id = len(bins)
            for idx in range(len(bins)):
                if freq <= bins[idx]:
                    bin_id = idx
                    break
            avg_scores[bin_id].append(avg_score)

    for bin_id, s in enumerate(avg_scores):
        means.append(np.mean(s))
    return means


means = calc_mean_loss(tokens, scores)
kmeans_means = calc_mean_loss(kmeans_tokens, kmeans_scores)

import matplotlib.pyplot as plt

X_axis = np.arange(len(x))
plt.bar(X_axis - 0.2, means, 0.4, label='original')
plt.bar(X_axis + 0.2, kmeans_means, 0.4, label='kmeans')

plt.xticks(X_axis, x)
plt.xlabel("Token Freq in Training Corpus")
plt.ylabel("Entropy Loss")
plt.legend()
plt.savefig('loss_per_word.png')
