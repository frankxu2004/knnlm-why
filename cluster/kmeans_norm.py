import faiss
import numpy as np

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
first_zero_idx = (vals == 0).argmax(axis=0)
if first_zero_idx == 0:
    # no zeros at all, all should be used
    first_zero_idx = len(vals)
to_cluster = np.zeros((first_zero_idx, vec_dim), dtype=np.float16)
to_cluster[:] = keys[:first_zero_idx]

# subsample for training kmeans
rs = np.random.RandomState(1)
idx = rs.choice(np.arange(first_zero_idx), size=int(0.2 * first_zero_idx), replace=False)
to_cluster = to_cluster[idx]
to_cluster = to_cluster.astype(np.float32)

print('Normalize training vectors')
faiss.normalize_L2(to_cluster)

print('start cluster')
ncentroids = dictionary.nspecial + ratio * (len(dictionary) - dictionary.nspecial)
niter = 20
verbose = True

kmeans = faiss.Kmeans(vec_dim, ncentroids, spherical=True, niter=niter, verbose=verbose, gpu=True, seed=1)
kmeans.train(to_cluster)

np.save(ckpt_path + 'norm_centroids.npy', kmeans.centroids)
