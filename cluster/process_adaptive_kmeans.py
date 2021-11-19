import faiss
import numpy as np
import torch
import csv
from tqdm import tqdm
import heapq
from scipy import sparse
from scipy.sparse import csr_matrix

from fairseq.data import Dictionary

dstore_size = 153225485
# dstore_size = 112733184

vec_dim = 1024
ratio = 2
ckpt_path = 'checkpoints/wikitext103-bpe/'
# ckpt_path = 'checkpoints/wikitext103-bpe/last_linear_inp/'

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')
print(len(dictionary))

prio_q = torch.load(ckpt_path + 'prio_q.pt')

centroids_by_word = {}

for score, word_id, centroids in prio_q:
    centroids_by_word[word_id] = (score, centroids)

all_vecs = []
all_vocab_ids = []
all_num_vecs = []
objectives = []

to_write = []
for i in tqdm(range(len(dictionary))):
    if i in centroids_by_word:
        centroids = centroids_by_word[i][1]
        all_vecs.append(centroids)
        all_vocab_ids.extend([i] * len(centroids))

        to_write.append((i, dictionary[i], dictionary.count[i], -centroids_by_word[i][0], len(centroids)))

with open('kmeans_adaptive_stats.csv', 'w') as out:
    csv_out = csv.writer(out)
    for row in to_write:
        csv_out.writerow(row)

all_vecs = np.concatenate(all_vecs)

print(all_vecs.shape)

assert len(all_vecs) == len(all_vocab_ids)

freq_mat = np.zeros((len(all_vecs), len(dictionary)))

for i in range(len(all_vecs)):
    freq_mat[i, all_vocab_ids[i]] = 1

freq_mat = csr_matrix(freq_mat)

sparse.save_npz(ckpt_path + 'kmeans_adaptive_freq.npz', freq_mat)

np.save(ckpt_path + 'kmeans_adaptive.npy', all_vecs)
