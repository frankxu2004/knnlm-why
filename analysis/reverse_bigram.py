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


def compute_reverse_bigram_entropy(token_ids: List[int]) -> Dict[str, float]:
    """Compute the reverse bigram entropy of a text."""
    cnts = {}
    for i in range(1, len(token_ids)):
        prev = dictionary[token_ids[i - 1]]
        word = dictionary[token_ids[i]]
        if word not in cnts:
            cnts[word] = {}
        cnts[word][prev] = cnts[word].get(prev, 0) + 1
    return dist_to_entropy(cnts)


# def test_bigram():
#     texts = ["i am sam . sam i am ."]
#     bigram_entropies = compute_reverse_bigram_entropy(texts)
#     expected_entropies = {"</s>": 0.0, ".": math.log(2.0), "am": 0.0, "i": math.log(2.0), "sam": math.log(2.0)}
#     if bigram_entropies != expected_entropies:
#         print("Error: expected", expected_entropies, "but got", bigram_entropies)
#
#
# test_bigram()

dev_bigram_entropies = compute_reverse_bigram_entropy(dev_tokens)
train_bigram_entropies = compute_reverse_bigram_entropy(train_tokens)

with open('dev_reverse_bigram_entropies.txt', 'w') as outfile:
    for word in token_list_to_show:
        outfile.write(f'{word}\t{dev_bigram_entropies[word]}\n')

with open('train_reverse_bigram_entropies.txt', 'w') as outfile:
    for word in token_list_to_show:
        outfile.write(f'{word}\t{train_bigram_entropies[word]}\n')
