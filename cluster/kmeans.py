import faiss
import numpy as np

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

first_zero_idx = (vals == 0).argmax(axis=0)
print(first_zero_idx)
to_cluster = np.zeros((first_zero_idx, vec_dim), dtype=np.float16)
to_cluster[:] = keys[:first_zero_idx]

# subsample for training kmeans
rs = np.random.RandomState(1)
idx = rs.choice(np.arange(first_zero_idx), size=int(0.2*first_zero_idx), replace=False)
to_cluster = to_cluster[idx]
to_cluster = to_cluster.astype(np.float32)

print('start cluster')
ncentroids = dictionary.nspecial + ratio * (len(dictionary) - dictionary.nspecial)
niter = 20
verbose = True

kmeans = faiss.Kmeans(vec_dim, ncentroids, niter=niter, verbose=verbose, gpu=True, seed=1)
kmeans.train(to_cluster)

np.save('checkpoints/wikitext103-bpe/centroids.npy', kmeans.centroids)
