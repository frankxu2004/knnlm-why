import numpy as np
from fairseq.data import Dictionary
from tqdm import tqdm

# dstore_size = 153225485
dstore_size = 112733184
# ckpt_path = 'checkpoints/wikitext103-bpe/'
ckpt_path = 'checkpoints/wikitext103-bpe/last_linear_inp/'

vec_dim = 1024

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')

keys_from_memmap = np.memmap(ckpt_path + 'dstore_keys.npy',
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

vals = vals[:first_zero_idx]

keys = np.zeros((first_zero_idx, vec_dim), dtype=np.float16)

keys[:] = keys_from_memmap[:first_zero_idx]
del keys_from_memmap


mean_vecs = []
for i in tqdm(range(len(dictionary))):
    vecs = keys[vals == i]
    if len(vecs):
        mean_vecs.append(np.mean(vecs, axis=0, dtype=np.float64))
    else:
        mean_vecs.append(np.zeros(vec_dim))

dstore_weight_matrix = np.stack(mean_vecs)
np.save(ckpt_path + 'centroid_word.npy', dstore_weight_matrix)
