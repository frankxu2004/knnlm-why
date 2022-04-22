import numpy as np

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

tokens = np.load('train_tokens.npy').astype(np.int32)
tgt_len = tokens.size
skipped_toks = 0
for i in range(tgt_len - 1):
    if tokens[i].item() in bpe_toks:
        skipped_toks += 1

count = len(tokens) - skipped_toks

token_freqs = [0] * len(dictionary)

for tid in tokens:
    token_freqs[tid] += 1

print('tok freq count done')


def process_scores(score_filename):
    print(score_filename)
    lm_scores = np.load(score_filename)
    total_loss = 0.

    token_scores = [0.] * len(dictionary)
    assert len(tokens) == len(lm_scores)

    for t, s in zip(tokens, lm_scores):
        token_scores[t] += -s
        total_loss += -s

    with open(score_filename + '.txt', 'w', encoding='utf-8') as outfile:
        for i in range(len(dictionary)):
            token_freq = token_freqs[i]
            sum_scores = token_scores[i]
            outfile.write(dictionary[i] + '\t' + str(token_freq) + '\t' + str(sum_scores) + '\n')

    print('train loss:', total_loss/count, 'ppl:', np.exp(total_loss/count))

process_scores('train_scores.npy')
process_scores('train_knn_scores.npy')
process_scores('overfit_train_scores.npy')
process_scores('overfit129_train_scores.npy')
