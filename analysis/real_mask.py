import numpy as np
import time
import torch
import torch.nn.functional as F
import tqdm


@torch.jit.script
def my_cdist(x1, x2, x2_norm):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-20)
    return res


topk = 1024
dimension = 1024
dstore_filename = "checkpoints/wikitext103-bpe/dstore_subsampled_0.05"
dstore_size = 7661274

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
keys = keys.astype(np.float16)

vals = vals_from_memmap[:]
vals = vals.astype(np.int64)

keys = torch.from_numpy(keys)
vals = torch.from_numpy(vals)

print('Loading to memory took {} s'.format(time.time() - start))

keys_norm = keys.cuda().pow_(2).sum(dim=-1, keepdim=True)
keys = keys.cuda()
vals = vals.cuda()
print(vals.dtype)
batch_size = 200
num_batches = len(queries) // batch_size + 1


for temp in np.arange(2.0, 15.1, 0.1):
    all_probs = []
    all_knns = []
    all_dists = []
    temp = float(temp)

    for batch_idx in tqdm.tqdm(range(num_batches)):
        batch_queries = queries[batch_idx * batch_size:(batch_idx + 1) * batch_size].cuda()
        tgt = tokens[batch_idx * batch_size:(batch_idx + 1) * batch_size].cuda()
        distances = -my_cdist(batch_queries, keys, keys_norm)
        res = distances.topk(topk, dim=1)
        knns = res.indices

        all_knns.append(knns.cpu().numpy())
        dists = res.values
        probs = F.log_softmax(dists/temp, dim=-1)
        index_mask = torch.eq(vals[knns].squeeze(-1), tgt.unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000  # for stability
        index_mask[index_mask == 1] = 0
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1)
        all_probs.append(yhat_knn_prob.cpu().numpy())
        all_dists.append(dists.sum(dim=-1).cpu().numpy())


    np.save(dstore_filename.split('/')[-1] + '_real_mask.npy' + str(temp), np.concatenate(all_probs))
    np.save(dstore_filename.split('/')[-1] + '_real_mask_knns.npy' + str(temp), np.concatenate(all_knns, axis=0))
    np.save(dstore_filename.split('/')[-1] + '_real_mask_dists.npy' + str(temp), np.concatenate(all_dists))

