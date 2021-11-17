# last_ffn_input
## eval - random sample 2 per word
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --load-centroids checkpoints/wikitext103-bpe/rsample.npy \
    --load-centroid-distribution checkpoints/wikitext103-bpe/rsample_freq.npz

# use_l2 random sample 2 per word
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True, 'use_l2': True}" \
    --load-centroids checkpoints/wikitext103-bpe/rsample.npy \
    --load-centroid-distribution checkpoints/wikitext103-bpe/rsample_freq.npz

