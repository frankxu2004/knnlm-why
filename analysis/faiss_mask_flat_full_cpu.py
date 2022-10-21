import numpy as np
import time
import tqdm
import faiss

topk = 1024
dimension = 1024
dstore_filename = "checkpoints/wikitext103-bpe/dstore"
dstore_size = 153225485

print('Loading queries...')
queries = np.load('all_queries.npy').astype(np.float32)

print(queries.dtype)

keys_from_memmap = np.memmap(dstore_filename + '_keys.npy', dtype=np.float16, mode='r',
                             shape=(dstore_size, dimension))
vals_from_memmap = np.memmap(dstore_filename + '_vals.npy', dtype=np.int64, mode='r',
                             shape=(dstore_size, 1))
print('Loading to memory...')
start = time.time()

keys = keys_from_memmap[:]
keys = keys.astype(np.float32)

vals = vals_from_memmap[:]
vals = vals.astype(np.int64)

print('Loading to memory took {} s'.format(time.time() - start))

index = faiss.IndexFlatL2(dimension)
index.add(keys)
print("FlatL2 index built!")

batch_size = 2000
num_batches = len(queries) // batch_size + 1

all_knns = []
all_dists = []

for batch_idx in tqdm.tqdm(range(num_batches)):
    batch_queries = queries[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    dists, knns = index.search(batch_queries, topk)
    all_knns.append(knns)
    dists = -1*dists
    all_dists.append(dists)

np.save(dstore_filename.split('/')[-1] + '_faiss_mask_flat_full_cpu_knns.npy', np.concatenate(all_knns, axis=0))
np.save(dstore_filename.split('/')[-1] + '_faiss_mask_flat_full_cpu_dists.npy', np.concatenate(all_dists, axis=0))
