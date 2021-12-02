import faiss
import numpy as np
from faiss import IndexFlatL2, IndexFlatIP

from fairseq.data import Dictionary

dstore_size = 153225485
# dstore_size = 112733184

vec_dim = 1024
ratio = 2
ckpt_path = 'checkpoints/wikitext103-bpe/'
# ckpt_path = 'checkpoints/wikitext103-bpe/last_linear_inp/'


dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')
print(len(dictionary))

keys = np.memmap(ckpt_path + 'dstore_keys.npy',
                 dtype=np.float16, mode='r', shape=(dstore_size, vec_dim))
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

print('to add:', first_zero_idx)

centroids = np.load(ckpt_path + 'norm_centroids.npy')
index = IndexFlatIP(vec_dim)
index = faiss.index_cpu_to_all_gpus(index)
index.add(centroids)

start = 0
num_keys_to_search_at_a_time = 500000

dists = []
centroid_ids = []

while start < first_zero_idx:
    end = min(first_zero_idx, start+num_keys_to_search_at_a_time)
    to_search = keys[start:end].copy()
    to_search = to_search.astype(np.float32)
    faiss.normalize_L2(to_search)
    d, i = index.search(to_search.astype(np.float32), 1)
    dists.append(d.squeeze())
    centroid_ids.append(i.squeeze())
    start += num_keys_to_search_at_a_time
    if (start % 1000000) == 0:
        print('Assigned %d tokens so far' % start)

print(np.concatenate(dists).shape)
np.save(ckpt_path + 'norm_dist.npy', np.concatenate(dists))
np.save(ckpt_path + 'norm_centroid_ids.npy', np.concatenate(centroid_ids))
