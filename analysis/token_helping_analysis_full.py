import math
import glob

import numpy as np
import torch

from fairseq.data import Dictionary

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')
print(len(dictionary))

tokens = np.load('tokens.npy')
lm_scores = np.load('scores.npy')

assert len(tokens) == len(lm_scores)

lm_scores = torch.from_numpy(lm_scores)

knn_helping = 0

score_files_dict = {
    'faiss_flat_mask': glob.glob('full_temp_exp/dstore_faiss_mask_flat_full.npy1.6.npy'),
    'faiss_mask': glob.glob('full_temp_exp/dstore_faiss_mask_full.npy0.9.npy'),
    'faiss_mask_recomp': glob.glob('full_temp_exp/dstore_faiss_mask_full_recomp.npy2.0.npy'),
}

for name in score_files_dict:
    extra_score_files = score_files_dict[name]
    for f in ['best_knn_only_scores.npy'] + extra_score_files:
        print(f)
        token_helped_counter = {}

        extra_scores = np.load(f)
        extra_scores = torch.from_numpy(extra_scores)
        combine_probs = torch.stack([lm_scores, extra_scores], dim=0)

        oracle_scores, argmaxs = torch.max(combine_probs, dim=0)

        if 'best_knn_only_scores' in f:
            temperature = 0
        else:
            temperature = float(f.split('.npy')[1])

        for idx, t in enumerate(tokens):
            word = dictionary[t]
            if word not in token_helped_counter:
                if argmaxs[idx] == 1:
                    token_helped_counter[word] = {'total': 1, 'helped': 1}
                else:
                    token_helped_counter[word] = {'total': 1, 'helped': 0}
            else:
                token_helped_counter[word]['total'] += 1
                if argmaxs[idx] == 1:
                    token_helped_counter[word]['helped'] += 1
        with open(f'{f}_token_helped.txt', 'w') as outfile:
            for word in token_helped_counter:
                outfile.write(f'{word}\t{token_helped_counter[word]["total"]}'
                              f'\t{token_helped_counter[word]["helped"]}'
                              f'\t{token_helped_counter[word]["helped"] / token_helped_counter[word]["total"]}\n')
