# k=1
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv1-interpolated \
  --arch transformer_lm_wikibpe  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --finetune-out-embed --init-out-embed --criterion agg_softmax --interpolated-loss \
  --no-epoch-checkpoints \
  --max-update 286000 --lr 1.0  --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
  --warmup-updates 3000 --warmup-init-lr 1e-7 --optimizer nag --clip-norm 1 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

## eval
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-kv1-interpolated/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --model-overrides "{'interpolated_loss': False}" \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --save-scores interpolated_loss_scores/kv1_interpolated_scores.npy

python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-kv1-interpolated-biglr/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --model-overrides "{'interpolated_loss': False}" \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --save-scores interpolated_loss_scores/kv1_biglr_interpolated_scores.npy

# att k=1
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv1-att-interpolated \
  --arch transformer_lm_wikibpe  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --use-last-ffn-input \
  --finetune-out-embed --init-out-embed --criterion agg_softmax --interpolated-loss \
  --no-epoch-checkpoints \
  --max-update 286000 --lr 1.0  --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
  --warmup-updates 3000 --warmup-init-lr 1e-7 --optimizer nag --clip-norm 1 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

## eval
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-kv1-att-interpolated/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --model-overrides "{'interpolated_loss': False}" \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --save-scores interpolated_loss_scores/kv1_att_interpolated_scores.npy


# ffn k=3
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv3-interpolated \
  --arch transformer_lm_wikibpe  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --finetune-out-embed --criterion agg_softmax --interpolated-loss \
  --pseudo-vocab-ratio 3 --no-epoch-checkpoints \
  --max-update 286000 --lr 1.0  --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
  --warmup-updates 3000 --warmup-init-lr 1e-7 --optimizer nag --clip-norm 1 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16


## eval
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-kv3-interpolated/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --model-overrides "{'interpolated_loss': False}" \
    --gen-subset valid --bpe subword_nmt --remove-bpe --pseudo-vocab-ratio 3 \
    --save-scores interpolated_loss_scores/kv3_interpolated_scores.npy

# att k=3
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv3-att-interpolated \
  --arch transformer_lm_wikibpe  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --use-last-ffn-input \
  --finetune-out-embed --criterion agg_softmax --interpolated-loss \
  --pseudo-vocab-ratio 3 --no-epoch-checkpoints \
  --max-update 286000 --lr 1.0  --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
  --warmup-updates 3000 --warmup-init-lr 1e-7 --optimizer nag --clip-norm 1 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

## eval
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-kv3-att-interpolated/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe --pseudo-vocab-ratio 3 \
    --model-overrides "{'interpolated_loss': False, 'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --save-scores interpolated_loss_scores/kv3_att_interpolated_scores.npy
