import faiss
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
import torch
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
for i in tqdm(range(len(dictionary))):
    vecs = np.load('dstore/ids/' + str(i) + '.npy')
    num_vecs = len(vecs)
    if num_vecs > 0:
        # subsample for training kmeans
        idx = rs.choice(np.arange(num_vecs), size=min(1000000, num_vecs), replace=False)
        to_cluster = vecs[idx]
        to_cluster = to_cluster.astype(np.float32)
        all_to_cluster[i] = to_cluster
        centroid = np.mean(to_cluster, axis=0)
        variance = np.sum((to_cluster - centroid) ** 2, axis=1).mean()
        centroid = centroid.reshape(1, -1)
        heapq.heappush(prio_q, (-variance, i, centroid))

# pick highest within-cluster variance
cluster_count = len(all_to_cluster)
print(cluster_count, 'added')

while cluster_count < threshold:
    if cluster_count % 1000 == 0:
        print('reached', cluster_count)
    _, w, old_clusters = heapq.heappop(prio_q)
    ncentroids = len(old_clusters) + 1
    to_cluster = all_to_cluster[w]
    num_to_cluster = len(to_cluster)
    if num_to_cluster < ncentroids * 40:
        # points too few, should not further cluster.
        heapq.heappush(prio_q, (9999, w, old_clusters))
        continue
    use_gpu = True
    if num_to_cluster < 10000:
        use_gpu = False
    kmeans = faiss.Kmeans(vec_dim, ncentroids, niter=20, verbose=False, gpu=use_gpu, seed=1)
    obj = kmeans.train(to_cluster)
    variance = obj / num_to_cluster
    heapq.heappush(prio_q, (-variance, w, kmeans.centroids))
    cluster_count += 1

torch.save(prio_q, ckpt_path + 'prio_q.pt')
