# finetune additional linear only
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-additional-linear \
  --arch transformer_lm_wikibpe --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --additional-linear --finetune-additional-linear \
  --max-update 28600 --optimizer nag --lr 1e-3 --clip-norm 1 \
  --max-tokens 3072 --update-freq 6 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

# continue finetuning
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-additional-linear \
  --arch transformer_lm_wikibpe --restore-file checkpoints/wikitext103-bpe-additional-linear/checkpoint_best.pt \
  --knn-keytype last_ffn_input --additional-linear --finetune-additional-linear \
  --max-update 28600 --optimizer nag --lr 1e-2 --clip-norm 0.1 \
  --max-tokens 18432 --update-freq 1 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16


# eval finetuned
python eval_lm.py data-bin/wikitext103-bpe --path checkpoints/wikitext103-bpe-additional-linear/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072  --context-window 2560 --softmax-batch 1024  \
    --gen-subset valid --bpe subword_nmt --remove-bpe  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}"


## after softmax
CUDA_VISIBLE_DEVICES=4,5,7 python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-additional-linear-after-softmax \
  --arch transformer_lm_wikibpe \
  --restore-file checkpoints/wikitext103-bpe-additional-linear-after-softmax/checkpoint_last.pt \
  --knn-keytype last_ffn_input --use-last-ffn-input --finetune-out-embed \
  --max-update 286000 --optimizer nag --lr 5e-2 --clip-norm 100 \
  --max-tokens 36864 --update-freq 1 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

# eval finetuned
python eval_lm.py data-bin/wikitext103-bpe --path checkpoints/wikitext103-bpe-additional-linear-after-softmax/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072  --context-window 2560 --softmax-batch 1024  \
    --gen-subset valid --bpe subword_nmt --remove-bpe  \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}"

## ATT K=1
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv1-att-fix \
  --arch transformer_lm_wikibpe --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --knn-keytype last_ffn_input --use-last-ffn-input --finetune-out-embed --init-out-embed \
  --reset-optimizer --reset-dataloader --reset-meters \
  --max-update 100000 --max-lr 1.0 --t-mult 2 --lr-scheduler cosine --lr-shrink 0.75 \
  --warmup-updates 5000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.01 --clip-norm 0.1 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

python eval_lm.py data-bin/wikitext103-bpe --path checkpoints/wikitext103-bpe-kv1-att-fix/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072  --context-window 2560 --softmax-batch 1024  \
    --gen-subset valid --bpe subword_nmt --remove-bpe  \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --save-scores kv1_att_finetune_scores.npy

## ATT K=9
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv9-att-fix \
  --arch transformer_lm_wikibpe \
  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --use-last-ffn-input --finetune-out-embed \
  --pseudo-vocab-ratio 9 --criterion agg_softmax \
  --max-update 286000 --optimizer nag --lr 5e-2 --clip-norm 100 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

python eval_lm.py data-bin/wikitext103-bpe --path checkpoints/wikitext103-bpe-kv9-att-fix/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072  --context-window 2560 --softmax-batch 1024  \
    --gen-subset valid --bpe subword_nmt --remove-bpe  --pseudo-vocab-ratio 9 \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --save-scores kv9_att_finetune_scores.npy

## ATT K=5
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv5-att-fix \
  --arch transformer_lm_wikibpe \
  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --use-last-ffn-input --finetune-out-embed \
  --pseudo-vocab-ratio 5 --criterion agg_softmax \
  --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
  --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

## ATT K=3
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv3-att-fix \
  --arch transformer_lm_wikibpe \
  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --use-last-ffn-input --finetune-out-embed \
  --pseudo-vocab-ratio 3 --criterion agg_softmax \
  --max-update 100000 --max-lr 0.5 --t-mult 2 --lr-scheduler cosine --lr-shrink 0.75 \
  --warmup-updates 5000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

## ATT K=6
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv6-att-fix \
  --arch transformer_lm_wikibpe \
  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --use-last-ffn-input --finetune-out-embed \
  --pseudo-vocab-ratio 6 --criterion agg_softmax \
  --max-update 100000 --max-lr 1.0 --t-mult 2 --lr-scheduler cosine --lr-shrink 0.75 \
  --warmup-updates 5000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.01 --clip-norm 0.1 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16


## ATT K=2
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv2-att-fix \
  --arch transformer_lm_wikibpe \
  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --use-last-ffn-input --finetune-out-embed \
  --pseudo-vocab-ratio 2 --criterion agg_softmax \
  --max-update 100000 --max-lr 1.0 --t-mult 2 --lr-scheduler cosine --lr-shrink 0.75 \
  --warmup-updates 5000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.01 --clip-norm 0.1 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

## ATT K=4
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv4-att-fix \
  --arch transformer_lm_wikibpe \
  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --use-last-ffn-input --finetune-out-embed \
  --pseudo-vocab-ratio 4 --criterion agg_softmax \
  --max-update 100000 --max-lr 1.0 --t-mult 2 --lr-scheduler cosine --lr-shrink 0.75 \
  --warmup-updates 5000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.01 --clip-norm 0.1 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

## ATT K=7
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv7-att-fix \
  --arch transformer_lm_wikibpe \
  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --use-last-ffn-input --finetune-out-embed \
  --pseudo-vocab-ratio 7 --criterion agg_softmax \
  --max-update 100000 --max-lr 1.0 --t-mult 2 --lr-scheduler cosine --lr-shrink 0.75 \
  --warmup-updates 10000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.01 --clip-norm 0.1 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

## ATT K=3 fixed lr
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv3-att-fix \
  --arch transformer_lm_wikibpe \
  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --use-last-ffn-input --finetune-out-embed \
  --pseudo-vocab-ratio 3 --criterion agg_softmax \
  --max-update 286000 --optimizer nag --lr 5e-2 --clip-norm 100 \
  --max-tokens 9216 --update-freq 1 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

## evaluate all
CUDA_VISIBLE_DEVICES=6 python eval_lm.py data-bin/wikitext103-bpe --path checkpoints/wikitext103-bpe-kv6-att-fix/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072  --context-window 2560 --softmax-batch 1024  \
    --gen-subset valid --bpe subword_nmt --remove-bpe  --pseudo-vocab-ratio 6 \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --save-scores kv6_att_finetune_scores.npy

