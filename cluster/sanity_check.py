import numpy as np
from fairseq.data import Dictionary
import tqdm
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse

dstore_size = 153225485
vec_dim = 1024

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')

keys_from_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_keys.npy',
                             dtype=np.float16, mode='r', shape=(dstore_size, vec_dim))

vals_from_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_vals.npy',
                             dtype=np.int, mode='r', shape=(dstore_size, 1))

vals = np.zeros((dstore_size, 1), dtype=np.int)

vals[:] = vals_from_memmap[:]
del vals_from_memmap

vals = vals.squeeze()

# after first zero it's all useless vecs
first_zero_idx = (vals == 0).argmax(axis=0)

vals = vals[:first_zero_idx]

