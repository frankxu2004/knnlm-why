import numpy as np

tokens = np.load('tokens.npy')

real_dists = np.load('dstore_subsampled_0.05_real_mask_dists.npy')
flat_dists = np.load('dstore_subsampled_0.05_faiss_mask_flat_dists.npy')
flat_cpu_dists = np.load('dstore_subsampled_0.05_faiss_mask_flat_cpu_dists.npy')

faiss_dists = np.load('dstore_subsampled_0.05_faiss_mask_dists.npy')
faiss_recomp_dists = np.load('dstore_subsampled_0.05_faiss_mask_recomp_dists.npy')
faiss_recomp_mycdist_dists = np.load('dstore_subsampled_0.05_faiss_mask_recomp_mycdist_dists.npy')


print('real', np.mean(real_dists)/1024)
print('faiss approx', np.mean(faiss_dists)/1024)
print('faiss approx recomp', np.mean(faiss_recomp_dists)/1024)
print('faiss approx recomp with my_cdist', np.mean(faiss_recomp_mycdist_dists)/1024)

print('faiss Flat GPU', np.mean(flat_dists)/1024)
print('faiss Flat CPU', np.mean(flat_cpu_dists)/1024)

print(np.mean(flat_cpu_dists/1024-flat_dists/1024))
print(np.std(flat_cpu_dists/1024-flat_dists/1024))

real_knns = np.load('dstore_subsampled_0.05_faiss_mask_flat_cpu_knns.npy')
faiss_knns = np.load('dstore_subsampled_0.05_faiss_mask_flat_knns.npy')

print(faiss_knns.shape)
print(real_knns.shape)
print(tokens.shape)

overlap = []
for idx in range(len(tokens)):
    overlap_number = len(set(faiss_knns[idx]).intersection(set(real_knns[idx])))
    overlap.append(overlap_number / 1024)

print(np.mean(overlap))
print(np.std(overlap))
