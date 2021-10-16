import numpy as np


dstore_size = 153225485
vec_dim = 1024

new_dstore_size = 112735285

keys_from_memmap = np.memmap('checkpoints/wikitext103-bpe-tied/dstore_keys.npy',
                             dtype=np.float16, mode='r', shape=(dstore_size, vec_dim))
vals_from_memmap = np.memmap('checkpoints/wikitext103-bpe-tied/dstore_vals.npy',
                             dtype=np.int, mode='r', shape=(dstore_size, 1))

new_keys_from_memmap = np.memmap('checkpoints/wikitext103-bpe-tied/new_dstore_keys.npy',
                             dtype=np.float16, mode='w+', shape=(new_dstore_size, vec_dim))
new_vals_from_memmap = np.memmap('checkpoints/wikitext103-bpe-tied/new_dstore_vals.npy',
                             dtype=np.int, mode='w+', shape=(new_dstore_size, 1))

new_keys_from_memmap[:] = keys_from_memmap[:new_dstore_size]
new_keys_from_memmap[:] = vals_from_memmap[:new_dstore_size]

del keys_from_memmap, vals_from_memmap, new_vals_from_memmap, new_vals_from_memmap