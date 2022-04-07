# k=2
CUDA_VISIBLE_DEVICES=4,5,6 python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv2-fix \
  --arch transformer_lm_wikibpe  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --finetune-out-embed --pseudo-vocab-ratio 2 --criterion agg_softmax \
  --max-update 28600 --optimizer nag --lr 1e-3 --clip-norm 100 \
  --max-tokens 3072 --update-freq 6 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16


## eval
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-kv2-fix/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --pseudo-vocab-ratio 2 \
    --gen-subset valid --bpe subword_nmt --remove-bpe

# k=3
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv3 \
  --arch transformer_lm_wikibpe \
  --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
  --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
  --max-tokens 3072 --update-freq 4 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16 \
  --pseudo-vocab-ratio 3 --criterion agg_softmax

## eval
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-kv3/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --pseudo-vocab-ratio 3 \
    --gen-subset valid --bpe subword_nmt --remove-bpe
