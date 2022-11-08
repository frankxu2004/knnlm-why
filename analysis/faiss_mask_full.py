import numpy as np
import time
import torch
import torch.nn.functional as F
import tqdm
import faiss


topk = 1024
dimension = 1024
dstore_filename = "checkpoints/wikitext103-bpe/dstore"
dstore_size = 153225485
indexfile = "checkpoints/wikitext103-bpe/knn.index"

index = faiss.read_index(indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
index.nprobe = 32

print('Loading tokens...')
tokens = np.load('tokens.npy')
tokens = torch.from_numpy(tokens)

print('Loading queries...')
queries = np.load('all_queries.npy').astype(np.float16)
queries = torch.from_numpy(queries)

assert len(queries) == len(tokens)
print(queries.dtype)

keys_from_memmap = np.memmap(dstore_filename + '_keys.npy', dtype=np.float16, mode='r',
                             shape=(dstore_size, dimension))
vals_from_memmap = np.memmap(dstore_filename + '_vals.npy', dtype=np.int64, mode='r',
                             shape=(dstore_size, 1))
print('Loading to memory...')
start = time.time()

keys = keys_from_memmap[:]

vals = vals_from_memmap[:]

# print('Loading to memory took {} s'.format(time.time() - start))

batch_size = 200
num_batches = len(queries) // batch_size + 1

all_knns = []
all_dists = []

for batch_idx in tqdm.tqdm(range(num_batches)):
    batch_queries = queries[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    tgt = tokens[batch_idx * batch_size:(batch_idx + 1) * batch_size].cuda()
    dists, knns = index.search(batch_queries.float().numpy(), topk)
    all_knns.append(knns)
    all_dists.append(dists)
    np.save(dstore_filename.split('/')[-1] + '_faiss_mask_full_knns.npy', np.concatenate(all_knns, axis=0))
    np.save(dstore_filename.split('/')[-1] + '_faiss_mask_full_dists.npy', np.concatenate(all_dists, axis=0))
