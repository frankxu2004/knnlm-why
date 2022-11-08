import numpy as np


def topk_by_partition(input, k, axis=None, ascending=True):
    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis)  # k non-sorted indices
    input = np.take_along_axis(input, ind, axis=axis)  # k non-sorted values

    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind_part, axis=axis)
    return ind, val


prefix_list = ['dstore_faiss_mask_flat_0_51075161',
               'dstore_faiss_mask_flat_51075161_102150322',
               'dstore_faiss_mask_flat_102150322_153225485']
all_knns = []
all_dists = []
for prefix in prefix_list:
    knns = np.load(f'{prefix}_knns.npy')
    dists = np.load(f'{prefix}_dists.npy')
    all_knns.append(knns + int(prefix.split('_')[-2]))
    all_dists.append(dists)

all_knns = np.concatenate(all_knns, axis=1)
all_dists = np.concatenate(all_dists, axis=1)


merged_knns, merged_dists = topk_by_partition(all_dists, 1024, axis=1, ascending=False)

merged_knns = np.take_along_axis(all_knns, merged_knns, axis=1)

print(merged_knns.shape)
print(merged_dists.shape)

np.save('dstore_faiss_mask_flat_full_knns.npy', merged_knns)
np.save('dstore_faiss_mask_flat_full_dists.npy', merged_dists)
