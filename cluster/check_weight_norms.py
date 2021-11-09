import numpy as np
from fairseq.data import Dictionary
import matplotlib.pyplot as plt

# ckpt_path = 'checkpoints/wikitext103-bpe/'
ckpt_path = 'checkpoints/wikitext103-bpe/last_linear_inp/'

dictionary = Dictionary.load('data-bin/wikitext103-bpe/dict.txt')

weight_matrix = np.load('checkpoints/wikitext103-bpe/out_embed.npy')

token_centroids = np.load(ckpt_path + 'centroid_word.npy')

kmeans_centroids = np.load(ckpt_path + 'centroids.npy')

out_emb_norms = np.linalg.norm(weight_matrix, axis=1)
token_centroids_norms = np.linalg.norm(token_centroids, axis=1)
kmeans_centroids_norms = np.linalg.norm(kmeans_centroids, axis=1)

print(token_centroids_norms.mean())
print(kmeans_centroids_norms.mean())
print(out_emb_norms.mean())

bins = np.linspace(0, 75, 100)

plt.hist([out_emb_norms, token_centroids_norms, kmeans_centroids_norms],
         bins=bins, label=['emb', 'centroid', 'kmeans'], density=True)
plt.legend(loc='upper right')
plt.title('last_linear_input')
plt.savefig('embedding_norms_last_linear_input.png')
