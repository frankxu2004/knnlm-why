# last_ffn_input
## eval - kmeans
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --load-centroids checkpoints/wikitext103-bpe/centroids.npy \
    --load-centroid-distribution checkpoints/wikitext103-bpe/cluster_freq.npz

## eval - spherical kmeans
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --load-centroids checkpoints/wikitext103-bpe/spherical_centroids.npy \
    --load-centroid-distribution checkpoints/wikitext103-bpe/spherical_cluster_freq.npz

## eval - normalized kmeans
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True, 'norm_l2': True}" \
    --load-centroids checkpoints/wikitext103-bpe/norm_centroids.npy \
    --load-centroid-distribution checkpoints/wikitext103-bpe/norm_cluster_freq.npz

## eval - kmeans with idw coef
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --load-centroids checkpoints/wikitext103-bpe/centroids.npy \
    --load-centroid-distribution checkpoints/wikitext103-bpe/cluster_freq_idw.npz

## eval - kmeans with softmax coef
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --load-centroids checkpoints/wikitext103-bpe/centroids.npy \
    --load-centroid-distribution checkpoints/wikitext103-bpe/cluster_freq_softmax.npz


## eval - word centroid
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --load-centroids checkpoints/wikitext103-bpe/centroid_word.npy

# use_l2 kmeans
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True, 'use_l2': True}" \
    --load-centroids checkpoints/wikitext103-bpe/centroids.npy \
    --load-centroid-distribution checkpoints/wikitext103-bpe/cluster_freq.npz

## use_l2 word centroid
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True, 'use_l2': True}" \
    --load-centroids checkpoints/wikitext103-bpe/centroid_word.npy

# last_linear_input
## eval - kmeans
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --model-overrides "{'use_last_ffn_input': False}" \
    --load-centroids checkpoints/wikitext103-bpe/last_linear_inp/centroids.npy \
    --load-centroid-distribution checkpoints/wikitext103-bpe/last_linear_inp/cluster_freq.npz

## eval - word centroid
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --model-overrides "{'use_last_ffn_input': False}" \
    --load-centroids checkpoints/wikitext103-bpe/last_linear_inp/centroid_word.npy
