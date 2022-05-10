import math
import json
import numpy as np

vocab_size = 0

log_base = 5

frequencies = []
with open('../train_scores.npy.txt') as loss_file:
    for line in loss_file:
        vocab_size += 1
        frequency = float(line.strip().split('\t')[2])
        print(frequency)
        frequencies.append(frequency)

print(vocab_size)

def additional_embeddings(log_base):
    num_extras = []
    for frequency in frequencies:
        if frequency != 0:
            num_additional_embeddings = math.floor(math.log(frequency, log_base))
            num_extras.append(num_additional_embeddings)
        else:
            num_extras.append(0)
    return num_extras

def total_embeddings(log_base):
    return vocab_size + sum(additional_embeddings(log_base))

# for base in np.linspace(12, 13, 100):
#     print(base)
#     print(total_embeddings(base) - 3 * vocab_size)
#
# exit()
best_base = 12.95959595959596 # 3V

print(total_embeddings(best_base)-3*vocab_size)

# save number of additional embeddings
json.dump(additional_embeddings(best_base), open('train_loss_num_extra_embed_3v.json', 'w'))

