import numpy as np

tokens = np.load('tokens.npy')

faiss_knns = np.load('dstore_subsampled_0.05_faiss_mask_flat_knns.npy')
real_knns = np.load('dstore_subsampled_0.05_real_mask_knns.npy')

print(faiss_knns.shape)
print(real_knns.shape)
print(tokens.shape)

matched = 0
for idx in range(len(tokens)):
    matched += len(set(faiss_knns[idx]).intersection(set(real_knns[idx]))) / 1024

print(matched)
print(matched/(len(tokens)))
