import numpy as np
token_centroids = np.load('checkpoints/wikitext103-bpe/centroid_word.npy')

print(token_centroids)
zero_idxs = [0, 1, 33341, 33342, 33343]

token_centroids[zero_idxs, :] = 0.

np.save('checkpoints/wikitext103-bpe/centroid_word.npy', token_centroids)