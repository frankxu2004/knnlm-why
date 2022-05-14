for i in {1..39}
do
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-att-extra-embed-3v-init-finetune/checkpoint${i}.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --context-window 2560 --softmax-batch 1024 --num-extra-embed-file analysis/train_freq_num_extra_embed_3v.json \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --save-scores epoch_scores/3v-att-init-finetune-epoch${i}.npy
done