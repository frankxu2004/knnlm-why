import numpy as np
from fairseq.data import Dictionary
import tqdm

dstore_size = 153225485

dists = np.load('checkpoints/wikitext103-bpe/dist.npy')
centroid_ids = np.load('checkpoints/wikitext103-bpe/centroid_ids.npy')
dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')

vals_from_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_vals.npy',
                             dtype=np.int, mode='r', shape=(dstore_size, 1))

vals = np.zeros((dstore_size, 1), dtype=np.int)

vals[:] = vals_from_memmap[:]
del vals_from_memmap

vals = vals.squeeze()

# after first zero it's all useless vecs
first_zero_idx = (vals == 0).argmax(axis=0)
vals = vals[:first_zero_idx]


freq_mat = np.zeros((np.max(centroid_ids)+1, len(dictionary)))

for centroid_id, word_id in tqdm.tqdm(zip(centroid_ids, vals)):
    freq_mat[centroid_id, word_id] += 1



