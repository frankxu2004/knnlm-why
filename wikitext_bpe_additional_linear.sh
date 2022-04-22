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


## after softmax, reinit
CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-additional-linear-after-softmax-reinit \
  --arch transformer_lm_wikibpe --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --knn-keytype last_ffn_input --use-last-ffn-input --finetune-out-embed --init-out-embed \
  --reset-optimizer --reset-dataloader --reset-meters \
  --max-update 286000 --optimizer nag --lr 1e-2 --clip-norm 100 \
  --max-tokens 12288 --update-freq 1 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16


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
