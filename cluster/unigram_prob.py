import numpy as np

from fairseq.data.data_utils import load_indexed_dataset
from fairseq.data import Dictionary

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')

train_dataset = load_indexed_dataset("data-bin/wikitext103-bpe/train", dictionary)

unigram_prob = np.zeros((len(dictionary), len(dictionary)))

for idx in range(len(train_dataset)-1):
    cur_token = train_dataset[idx][0].item()
    next_token = train_dataset[idx+1][0].item()
    unigram_prob[cur_token, next_token] += 1

print('normalizing')

row_sums = np.sum(unigram_prob, axis=1, keepdims=True) + 1e-5
unigram_prob = unigram_prob / row_sums

np.save('unigram_prob.npy', unigram_prob)
