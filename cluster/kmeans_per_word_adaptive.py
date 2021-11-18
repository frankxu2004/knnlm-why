import faiss
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from tqdm import tqdm
import heapq

from fairseq.data import Dictionary

dstore_size = 153225485
# dstore_size = 112733184

vec_dim = 1024
ratio = 2
ckpt_path = 'checkpoints/wikitext103-bpe/'
# ckpt_path = 'checkpoints/wikitext103-bpe/last_linear_inp/'

rs = np.random.RandomState(1)

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')
print(len(dictionary))
threshold = ratio * len(dictionary)

all_vecs = []
all_vocab_ids = []
all_num_vecs = []
objectives = []

prio_q = []

all_to_cluster = {}
# load data, initialize with 1 centroids
for i in tqdm(range(33300, len(dictionary))):
    vecs = np.load('dstore/ids/' + str(i) + '.npy')
    num_vecs = len(vecs)
    if num_vecs > 0:
        # subsample for training kmeans
        idx = rs.choice(np.arange(num_vecs), size=min(1000000, num_vecs), replace=False)
        to_cluster = vecs[idx]
        to_cluster = to_cluster.astype(np.float32)
        num_to_cluster = len(to_cluster)
        all_to_cluster[i] = to_cluster
        print(len(to_cluster))
        centroid = np.mean(to_cluster, axis=0)
        print(to_cluster - centroid)
        variance = np.abs(to_cluster - centroid) **2
        print(variance.shape)
        exit()
        ncentroids = 1
        use_gpu = True
        if num_to_cluster < 10000:
            use_gpu = False
        kmeans = faiss.Kmeans(vec_dim, ncentroids, niter=10, verbose=False, gpu=use_gpu, seed=1)
        obj = kmeans.train(to_cluster)
        print(kmeans.centroids)
        print(obj/num_to_cluster)
        objectives.append(obj)
        all_vecs.append(kmeans.centroids)
        all_vocab_ids.extend([i]*2)

all_vecs = np.concatenate(all_vecs)
objectives = np.array(objectives)
print(all_vecs.shape)

assert len(all_vecs) == len(all_vocab_ids)

freq_mat = np.zeros((len(all_vecs), len(dictionary)))

for i in range(len(all_vecs)):
    freq_mat[i, all_vocab_ids[i]] = 1


freq_mat = csr_matrix(freq_mat)

sparse.save_npz(ckpt_path + 'kmeans_2perword_freq.npz', freq_mat)

np.save(ckpt_path + 'kmeans_2perword.npy', all_vecs)
np.save(ckpt_path + 'kmeans_2perword_obj.npy', objectives)
