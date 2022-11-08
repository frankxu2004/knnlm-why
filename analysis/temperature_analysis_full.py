import math
import glob

import numpy as np
import torch

from fairseq.data import Dictionary

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')
print(len(dictionary))

bpe_cont = "@@"
bpe_toks = {
    i
    for i in range(len(dictionary))
    if dictionary[i].endswith(bpe_cont)
}

bpe_len = len(bpe_cont)

tokens = np.load('tokens.npy')
lm_scores = np.load('scores.npy')

assert len(tokens) == len(lm_scores)

lm_scores = torch.from_numpy(lm_scores).cuda()
tgt_len = tokens.size
skipped_toks = 0
for i in range(tgt_len - 1):
    if tokens[i].item() in bpe_toks:
        skipped_toks += 1

count = len(tokens) - skipped_toks

knn_helping = 0

score_files_dict = {
    'faiss_flat_mask': glob.glob('full_temp_exp/dstore_faiss_mask_flat_full.npy*.npy'),
    'faiss_mask': glob.glob('full_temp_exp/dstore_faiss_mask_full.npy*.npy'),
    'faiss_mask_recomp': glob.glob('full_temp_exp/dstore_faiss_mask_full_recomp.npy*.npy'),
}

for name in score_files_dict:
    extra_score_files = score_files_dict[name]
    with open(f'{name}_softmax_temperature_full.txt', 'w') as outfile:
        for f in ['best_knn_only_scores.npy'] + extra_score_files:
            print(f)
            extra_scores = np.load(f)
            extra_scores = torch.from_numpy(extra_scores).cuda()
            combine_probs = torch.stack([lm_scores, extra_scores], dim=0)

            oracle_scores, argmaxs = torch.max(combine_probs, dim=0)

            oracle_ppl = torch.exp(-oracle_scores.sum() / count)

            if 'best_knn_only_scores' in f:
                knn_helping = argmaxs
                temperature = 0
            else:
                temperature = float(f.split('.npy')[1])

            match_knn = torch.sum(argmaxs == knn_helping).item() / len(tokens)
            extra_helping_percentage = torch.sum(argmaxs).item() / len(tokens)

            knn_helping_scores = -(combine_probs[0][knn_helping == 0].sum() +
                                   combine_probs[1][knn_helping == 1].sum())

            knn_helping_ppl = torch.exp(knn_helping_scores / count)

            extra_only_ppl = torch.exp(-extra_scores.sum() / count)

            best_ppl = 1e10
            best_lmbda = 0
            for lmbda in np.linspace(0.001, 0.999, num=200):
                coeffs = torch.ones_like(combine_probs)
                coeffs[0] = np.log(1 - lmbda)
                coeffs[1] = np.log(lmbda)

                scores = torch.logsumexp(combine_probs + coeffs, dim=0)

                score_sum = scores.sum()

                avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
                ppl = 2 ** avg_nll_loss.item()
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_lmbda = lmbda

            outfile.write(f'{f}\t{temperature}\t{extra_only_ppl}\t{best_lmbda}\t{best_ppl}\t{oracle_ppl}\t'
                          f'{match_knn}\t{extra_helping_percentage}\t{knn_helping_ppl}\n')
