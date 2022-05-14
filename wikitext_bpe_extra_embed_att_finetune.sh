# train freq, approx 3V, att, all init
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-att-extra-embed-3v-init-finetune \
  --arch transformer_lm_wikibpe  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --use-last-ffn-input \
  --num-extra-embed-file analysis/train_freq_num_extra_embed_3v.json \
  --finetune-out-embed --criterion agg_softmax \
  --max-update 286000 --optimizer nag --lr 1e-1 --clip-norm 100 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

# train loss, approx 3V, att, all init
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-att-extra-embed-loss-3v-init-finetune \
  --arch transformer_lm_wikibpe  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --use-last-ffn-input \
  --num-extra-embed-file analysis/train_loss_num_extra_embed_3v.json \
  --finetune-out-embed --criterion agg_softmax \
  --max-update 286000 --optimizer nag --lr 5e-1 --clip-norm 100 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

## eval
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-att-extra-embed-3v-init-finetune/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --context-window 2560 --softmax-batch 1024 --num-extra-embed-file analysis/train_freq_num_extra_embed_3v.json \
    --gen-subset valid --bpe subword_nmt --remove-bpe --save-scores extra_embed_scores/3v-att-init-finetune.npy

python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-att-extra-embed-loss-3v-init-finetune/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --context-window 2560 --softmax-batch 1024 --num-extra-embed-file analysis/train_loss_num_extra_embed_3v.json \
    --gen-subset valid --bpe subword_nmt --remove-bpe --save-scores extra_embed_scores/loss-3v-att-init-finetune.npy


