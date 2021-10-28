import faiss
import numpy as np
from faiss import IndexFlatL2

from fairseq.data import Dictionary

dstore_size = 153225485
vec_dim = 1024
ratio = 2

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')
print(len(dictionary))

keys = np.memmap('checkpoints/wikitext103-bpe/dstore_keys.npy',
                             dtype=np.float16, mode='r', shape=(dstore_size, vec_dim))
vals_from_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_vals.npy',
                             dtype=np.int, mode='r', shape=(dstore_size, 1))

vals = np.zeros((dstore_size, 1), dtype=np.int)

vals[:] = vals_from_memmap[:]
del vals_from_memmap

vals = vals.squeeze()

# after first zero it's all useless vecs
first_zero_idx = (vals == 0).argmax(axis=0)
print('to add:', first_zero_idx)

centroids = np.load('checkpoints/wikitext103-bpe/centroids.npy')
index = IndexFlatL2(vec_dim)
index = faiss.index_cpu_to_all_gpus(index)
index.add(centroids)

start = 0
num_keys_to_search_at_a_time = 500000

dists = []
centroid_ids = []

while start < first_zero_idx:
    end = min(first_zero_idx, start+num_keys_to_search_at_a_time)
    to_search = keys[start:end].copy()
    d, i = index.search(to_search.astype(np.float32), 1)
    dists.append(d.squeeze())
    centroid_ids.append(i.squeeze())
    start += num_keys_to_search_at_a_time
    if (start % 1000000) == 0:
        print('Assigned %d tokens so far' % start)

print(np.concatenate(dists).shape)
np.save('checkpoints/wikitext103-bpe/dist.npy', np.concatenate(dists))
np.save('checkpoints/wikitext103-bpe/centroid_ids.npy', np.concatenate(centroid_ids))
