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
vals = np.memmap('checkpoints/wikitext103-bpe/dstore_vals.npy',
                             dtype=np.int, mode='r', shape=(dstore_size, 1))

vals = vals.squeeze()

to_cluster = keys[vals>=dictionary.nspecial]
np.save('checkpoints/wikitext103-bpe/to_cluster.npy', to_cluster)
to_cluster = to_cluster.astype(np.float32)

ncentroids = dictionary.nspecial + ratio * (len(dictionary) - dictionary.nspecial)
niter = 20
verbose = True

kmeans = faiss.Kmeans(vec_dim, ncentroids, niter=niter, verbose=verbose, gpu=True, seed=1)
kmeans.train(to_cluster)

np.save('checkpoints/wikitext103-bpe/centroids.npy', kmeans.centroids)

D, I = kmeans.index.search(to_cluster, 1)

np.save('checkpoints/wikitext103-bpe/dist.npy', D)
np.save('checkpoints/wikitext103-bpe/centroid_ids.npy', I)
