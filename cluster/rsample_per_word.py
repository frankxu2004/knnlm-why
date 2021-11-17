import faiss
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from tqdm import tqdm

from fairseq.data import Dictionary

dstore_size = 153225485
# dstore_size = 112733184

vec_dim = 1024
sample_per_word = 2
ckpt_path = 'checkpoints/wikitext103-bpe/'
# ckpt_path = 'checkpoints/wikitext103-bpe/last_linear_inp/'

rs = np.random.RandomState(1)

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

vals = vals[:first_zero_idx]

all_vecs = []
all_vocab_ids = []
for i in tqdm(range(len(dictionary))):
    idxes = np.nonzero(vals == i)[0]
    if len(idxes) > 0:
        idxes = rs.choice(idxes, sample_per_word, replace=False)
        vecs = keys[idxes]
        all_vecs.append(vecs)
        all_vocab_ids.extend([i]*sample_per_word)

all_vecs = np.concatenate(all_vecs)

print(all_vecs.shape)

assert len(all_vecs) == len(all_vocab_ids)

freq_mat = np.zeros((len(all_vecs), len(dictionary)))

for i in range(len(all_vecs)):
    freq_mat[i, all_vocab_ids[i]] = 1


freq_mat = csr_matrix(freq_mat)

sparse.save_npz(ckpt_path + 'rsample_freq.npz', freq_mat)

np.save(ckpt_path + 'rsample.npy', all_vecs)
