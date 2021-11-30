import torch
from pycave.bayes import GaussianMixture
import numpy as np

from fairseq.data import Dictionary

dstore_size = 153225485
# dstore_size = 112733184

vec_dim = 1024
ratio = 2
ckpt_path = 'checkpoints/wikitext103-bpe/'
# ckpt_path = 'checkpoints/wikitext103-bpe/last_linear_inp/'


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

# subsample for training GMM
rs = np.random.RandomState(1)
idx = rs.choice(np.arange(first_zero_idx), size=int(0.001 * first_zero_idx), replace=False)
to_cluster = keys[idx]
to_cluster = to_cluster.astype(np.float32)
ncomponents = dictionary.nspecial + ratio * (len(dictionary) - dictionary.nspecial)

print('Num Train:', len(to_cluster))

# load kmeans centroids for initialization
kmeans_centroids = np.load(ckpt_path + 'centroids.npy')

assert kmeans_centroids.shape[0] == ncomponents

kmeans_centroids = torch.from_numpy(kmeans_centroids)
to_cluster = torch.from_numpy(to_cluster)


estimator = GaussianMixture(num_components=ncomponents, init_means=kmeans_centroids, batch_size=20480,
                            trainer_params=dict(gpus=1), covariance_regularization=1e-4)
estimator.fit(to_cluster)

# Once the estimator is fitted, it provides various properties. One of them is
# the `model_` property which yields the PyTorch module with the fitted parameters.
print(estimator.nll_)
estimator.save(ckpt_path + 'gmm/')
