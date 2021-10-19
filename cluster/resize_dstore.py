import numpy as np


dstore_size = 153225485
vec_dim = 1024

# new_dstore_size = 112735285

keys_from_memmap = np.memmap('checkpoints/wikitext103-bpe-tied/dstore_keys.npy',
                             dtype=np.float16, mode='r', shape=(dstore_size, vec_dim))
vals_from_memmap = np.memmap('checkpoints/wikitext103-bpe-tied/dstore_vals.npy',
                             dtype=np.int, mode='r', shape=(dstore_size, 1))
#
# print(keys_from_memmap[0])
# print(vals_from_memmap[0])