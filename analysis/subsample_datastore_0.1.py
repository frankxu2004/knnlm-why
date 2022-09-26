import numpy as np

from fairseq.data import Dictionary
np.random.seed(0)
dstore_size = 153225485
vec_dim = 1024
subsample_size = 15322548

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')
print(len(dictionary))

keys_from_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_keys.npy',
                             dtype=np.float16, mode='r', shape=(dstore_size, vec_dim))
vals_from_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_vals.npy',
                             dtype=np.int64, mode='r', shape=(dstore_size, 1))

subsampled_idxs = np.random.choice(dstore_size, subsample_size, replace=False)

print('sampled idx created')
subsampled_keys_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_subsampled_0.1_keys.npy',
                                   dtype=np.float16, mode='w+', shape=(subsample_size, vec_dim))
subsampled_vals_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_subsampled_0.1_vals.npy',
                                   dtype=np.int64, mode='w+', shape=(subsample_size, 1))

subsampled_keys_memmap[:] = keys_from_memmap[subsampled_idxs]
subsampled_vals_memmap[:] = vals_from_memmap[subsampled_idxs]
