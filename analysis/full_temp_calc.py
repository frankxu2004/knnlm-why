import numpy as np
import time
import torch
import torch.nn.functional as F
import tqdm

topk = 1024
dimension = 1024
dstore_filename = "checkpoints/wikitext103-bpe/dstore"
dstore_size = 153225485

print('Loading tokens...')
tokens = np.load('tokens.npy')
tokens = torch.from_numpy(tokens)

print('Loading knns and dists')
# prefix = 'dstore_faiss_mask_flat_full'
# prefix = 'dstore_faiss_mask_full_recomp'
prefix = 'dstore_faiss_mask_full'

print(prefix)

knns_filename = prefix + '_knns.npy'
dists_filename = prefix + '_dists.npy'

all_knns = np.load(knns_filename)
all_dists = np.load(dists_filename).astype(np.float32)

assert len(all_knns) == len(tokens)

vals_from_memmap = np.memmap(dstore_filename + '_vals.npy', dtype=np.int64, mode='r',
                             shape=(dstore_size, 1))
print('Loading to memory...')
start = time.time()

vals = vals_from_memmap[:]
vals = vals.astype(np.int64)

print('Loading to memory took {} s'.format(time.time() - start))

batch_size = 2000000
num_batches = len(tokens) // batch_size + 1

for temp in tqdm.tqdm(np.arange(0.1, 3.1, 0.1)):
    all_probs = []
    for batch_idx in range(num_batches):
        dists = all_dists[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        knns = all_knns[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        tgt = tokens[batch_idx * batch_size:(batch_idx + 1) * batch_size].cuda()
        dists = torch.from_numpy(dists).cuda()
        probs = F.log_softmax(-dists/temp, dim=-1)
        index_mask = torch.eq(torch.from_numpy(vals[knns]).long().cuda().squeeze(-1), tgt.unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000  # for stability
        index_mask[index_mask == 1] = 0
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1)
        all_probs.append(yhat_knn_prob.cpu().numpy())

    np.save('full_temp_exp/' + knns_filename.replace('_knns', '') + str(temp),
            np.concatenate(all_probs))
