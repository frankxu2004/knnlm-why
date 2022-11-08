import numpy as np

num_splits = 2
dstore_size = 153225485
vec_dim = 1024

keys_from_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_keys.npy',
                             dtype=np.float16, mode='r', shape=(dstore_size, vec_dim))
vals_from_memmap = np.memmap('checkpoints/wikitext103-bpe/dstore_vals.npy',
                             dtype=np.int64, mode='r', shape=(dstore_size, 1))

split_size = dstore_size // num_splits
for i in range(num_splits):
    print(i)
    start = i * split_size
    if i == num_splits - 1:
        end = dstore_size
    else:
        end = (i + 1) * split_size

    current_sub_size = end - start

    subsampled_keys_memmap = np.memmap(f'checkpoints/wikitext103-bpe/dstore_subsampled_{start}_{end}_keys.npy',
                                       dtype=np.float16, mode='w+', shape=(current_sub_size, vec_dim))
    subsampled_vals_memmap = np.memmap(f'checkpoints/wikitext103-bpe/dstore_subsampled_{start}_{end}_vals.npy',
                                       dtype=np.int64, mode='w+', shape=(current_sub_size, 1))

    subsampled_keys_memmap[:] = keys_from_memmap[start:end]
    subsampled_vals_memmap[:] = vals_from_memmap[start:end]
