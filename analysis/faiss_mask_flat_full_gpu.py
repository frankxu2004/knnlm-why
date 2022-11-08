import numpy as np
import time
import tqdm
import faiss

num_splits = 3
dstore_size = 153225485
topk = 1024
dimension = 1024
dstore_filename = "checkpoints/wikitext103-bpe/dstore"

print('Loading queries...')
queries = np.load('all_queries.npy').astype(np.float32)

print(queries.dtype)

split_size = dstore_size // num_splits
for i in range(2, num_splits):
    print(i)
    keys_from_memmap = np.memmap(dstore_filename + '_keys.npy', dtype=np.float16, mode='r',
                                 shape=(dstore_size, dimension))
    vals_from_memmap = np.memmap(dstore_filename + '_vals.npy', dtype=np.int64, mode='r',
                                 shape=(dstore_size, 1))

    start = i * split_size
    if i == num_splits - 1:
        end = dstore_size
    else:
        end = (i + 1) * split_size

    current_sub_size = end - start

    keys = keys_from_memmap[start:end].astype(np.float32)
    vals = vals_from_memmap[start:end].astype(np.int64)

    print("Loaded to memory")

    index_cpu = faiss.IndexFlatL2(dimension)
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    index = faiss.index_cpu_to_all_gpus(index_cpu, co)
    index.add(keys)

    print("FlatL2 index built on GPU")

    batch_size = 2000
    num_batches = len(queries) // batch_size + 1

    all_knns = []
    all_dists = []

    for batch_idx in tqdm.tqdm(range(num_batches)):
        batch_queries = queries[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        dists, knns = index.search(batch_queries, topk)
        all_knns.append(knns)
        dists = -1 * dists
        all_dists.append(dists)

    np.save(dstore_filename.split('/')[-1] + f'_faiss_mask_flat_{start}_{end}_knns.npy',
            np.concatenate(all_knns, axis=0))
    np.save(dstore_filename.split('/')[-1] + f'_faiss_mask_flat_{start}_{end}_dists.npy',
            np.concatenate(all_dists, axis=0))

    break