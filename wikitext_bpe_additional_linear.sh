# finetune additional linear only
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-additional-linear \
  --arch transformer_lm_wikibpe --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --knn-keytype last_ffn_input --additional-linear \
  --max-update 28600 --optimizer nag --lr 1e-3 --clip-norm 1 \
  --max-tokens 3072 --update-freq 6 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

# continue finetuning
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-additional-linear \
  --arch transformer_lm_wikibpe --restore-file checkpoints/wikitext103-bpe-additional-linear/checkpoint_best.pt \
  --knn-keytype last_ffn_input --additional-linear \
  --max-update 28600 --optimizer nag --lr 1e-3 --clip-norm 1 \
  --max-tokens 9216 --update-freq 2 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16


# eval finetuned
python eval_lm.py data-bin/wikitext103-bpe --path checkpoints/wikitext103-bpe-additional-linear/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072  --context-window 2560 --softmax-batch 1024  \
    --gen-subset valid --bpe subword_nmt --remove-bpe  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}"
