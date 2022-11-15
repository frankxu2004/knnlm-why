import numpy as np
import math

from typing import Dict, List
from fairseq.data import Dictionary
from fairseq.data.data_utils import load_indexed_dataset

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')

train_dataset = load_indexed_dataset("data-bin/wikitext103-bpe/train", dictionary)

train_tokens = []

for sent in train_dataset:
    for token in sent:
        train_tokens.append(token.item())

train_tokens = np.array(train_tokens)
print(train_tokens.shape)
dev_tokens = np.load('tokens.npy')
print(dev_tokens.shape)
token_list_to_show = []
for idx, t in enumerate(dev_tokens):
    word = dictionary[t]
    if word not in token_list_to_show:
        token_list_to_show.append(word)


def dist_to_entropy(dists: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Convert a distribution to an entropy dictionary."""
    entropy = {}
    for word, dist in dists.items():
        entropy[word] = 0
        denom = sum(dist.values())
        for cnt in dist.values():
            p = cnt / denom
            entropy[word] += -p * math.log(p)
    return entropy


def compute_forward_bigram_entropy(token_ids: List[int]) -> Dict[str, float]:
    """Compute the forward bigram entropy of a text."""
    cnts = {}
    for i in range(0, len(token_ids)-1):
        word = dictionary[token_ids[i]]
        next_word = dictionary[token_ids[i+1]]
        if word not in cnts:
            cnts[word] = {}
        cnts[word][next_word] = cnts[word].get(next_word, 0) + 1
    return dist_to_entropy(cnts)

dev_bigram_entropies = compute_forward_bigram_entropy(dev_tokens)
train_bigram_entropies = compute_forward_bigram_entropy(train_tokens)

with open('dev_forward_bigram_entropies.txt', 'w') as outfile:
    for word in token_list_to_show:
        outfile.write(f'{word}\t{dev_bigram_entropies[word]}\n')

with open('train_forward_bigram_entropies.txt', 'w') as outfile:
    for word in token_list_to_show:
        outfile.write(f'{word}\t{train_bigram_entropies[word]}\n')
