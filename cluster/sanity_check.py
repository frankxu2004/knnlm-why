import pypeln as pl
import numpy as np
from fairseq.data import Dictionary
from tqdm import tqdm

dstore_size = 153225485
vec_dim = 1024

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')

weight_matrix = np.load('checkpoints/wikitext103-bpe/out_embed.npy')

keys_from_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_keys.npy',
                dtype=np.float16, mode='r', shape=(dstore_size, vec_dim))
vals_from_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_vals.npy',
                             dtype=np.int64, mode='r', shape=(dstore_size, 1))

vals = np.zeros((dstore_size, 1), dtype=np.int64)

vals[:] = vals_from_memmap[:]
del vals_from_memmap

vals = vals.squeeze()

first_zero_idx = (vals == 0).argmax(axis=0)

keys = np.zeros((first_zero_idx, vec_dim), dtype=np.float16)

keys[:] = keys_from_memmap[:first_zero_idx]
del keys_from_memmap

def calc_mean(word_id):
    vecs = np.load('dstore/ids/' + str(word_id) + '.npy')
    if len(vecs):
        return np.mean(vecs, axis=0, dtype=np.float64)
    else:
        return np.zeros(vec_dim)

mean_vecs = []
for i in tqdm(range(len(dictionary))):
    vecs = keys[vals==i]
    if vecs:
        mean_vecs.append(np.mean(vecs, axis=0, dtype=np.float64))
    else:
        mean_vecs.append(weight_matrix[i])

dstore_weight_matrix = np.stack(mean_vecs)
np.save('checkpoints/wikitext103-bpe/centroid_word.npy', dstore_weight_matrix)
