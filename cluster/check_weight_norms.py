import numpy as np
from fairseq.data import Dictionary
import matplotlib.pyplot as plt


dstore_size = 153225485
vec_dim = 1024

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')

weight_matrix = np.load('checkpoints/wikitext103-bpe/out_embed.npy')

token_centroids = np.load('checkpoints/wikitext103-bpe/centroid_word.npy')

kmeans_centroids = np.load('checkpoints/wikitext103-bpe/centroids.npy')

out_emb_norms = np.linalg.norm(weight_matrix, axis=1)
token_centroids_norms = np.linalg.norm(token_centroids, axis=1)
kmeans_centroids_norms = np.linalg.norm(kmeans_centroids, axis=1)

bins = np.linspace(0, 7, 100)

plt.hist([out_emb_norms, token_centroids_norms, kmeans_centroids_norms], bins=bins, label=['emb', 'centroid', 'kmeans'], density=True)
plt.legend(loc='upper right')
plt.savefig('embedding_norms.png')