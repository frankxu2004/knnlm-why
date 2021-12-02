import numpy as np
from fairseq.data import Dictionary
import tqdm
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse

dstore_size = 153225485
# dstore_size = 112733184
ckpt_path = 'checkpoints/wikitext103-bpe/'
# ckpt_path = 'checkpoints/wikitext103-bpe/last_linear_inp/'

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')

dists = np.load(ckpt_path + 'dist.npy')
centroid_ids = np.load(ckpt_path + 'centroid_ids.npy')

vals_from_memmap = np.memmap(ckpt_path + 'dstore_vals.npy',
                             dtype=np.int64, mode='r', shape=(dstore_size, 1))

vals = np.zeros((dstore_size, 1), dtype=np.int64)

vals[:] = vals_from_memmap[:]
del vals_from_memmap

vals = vals.squeeze()

# after first zero it's all useless vecs
first_zero_idx = (vals == 0).argmax(axis=0)
if first_zero_idx == 0:
    # no zeros at all, all should be used
    first_zero_idx = len(vals)

vals = vals[:first_zero_idx]

freq_mat = np.zeros((np.max(centroid_ids) + 1, len(dictionary)))

# consider distance to centroid when counting frequency
for centroid_id, word_id, dist in tqdm.tqdm(zip(centroid_ids, vals, dists)):
    freq_mat[centroid_id, word_id] += 1 / (dist + 1e-2)

freq_mat = csr_matrix(freq_mat)

sparse.save_npz(ckpt_path + 'cluster_count_idw.npz', freq_mat)

freq_mat = sparse.load_npz(ckpt_path + 'cluster_count_idw.npz')

print('Sparsity:', 1 - freq_mat.getnnz() / np.prod(freq_mat.shape))

cluster_size = csr_matrix.sum(freq_mat, axis=1)
cluster_size = np.squeeze(np.asarray(cluster_size), axis=1)

sums = np.repeat(cluster_size, freq_mat.getnnz(axis=1))
freq_mat.data /= sums

sparse.save_npz(ckpt_path + 'cluster_freq_idw.npz', freq_mat)

k = 10

data = []
for idx in tqdm.tqdm(range(freq_mat.shape[0])):
    row = freq_mat.getrow(idx).toarray()[0].ravel()
    top_k_indices = row.argsort()[-k:][::-1]
    top_k_values = row[top_k_indices]
    item = {'cluster_id': idx,
            'count': cluster_size[idx],
            }
    for j in range(k):
        assert top_k_indices[j] < len(dictionary)
        item['word'+str(j)] = dictionary[top_k_indices[j]]
        item['freq'+str(j)] = top_k_values[j]

    data.append(item)

df = pd.DataFrame(data)
df.to_csv(ckpt_path + 'cluster_idw.csv')

