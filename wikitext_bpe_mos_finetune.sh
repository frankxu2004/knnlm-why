# MOS k=3
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-mos3-finetune \
  --arch transformer_lm_wikibpe  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --finetune-mos --k-mos 3 \
  --max-update 286000 --optimizer nag --lr 5e-2 --clip-norm 100 \
  --max-tokens 9216 --update-freq 1 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16


## eval
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-mos3-finetune/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --k-mos 3 \
    --gen-subset valid --bpe subword_nmt --remove-bpe --save-scores mos_scores/mos3_finetune.npy


python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-mos2-finetune/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --k-mos 2 \
    --gen-subset valid --bpe subword_nmt --remove-bpe --save-scores mos_scores/mos2_finetune.npy

python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-mos4-finetune/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --k-mos 4 \
    --gen-subset valid --bpe subword_nmt --remove-bpe --save-scores mos_scores/mos4_finetune.npy

python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-mos5-finetune/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --k-mos 5 \
    --gen-subset valid --bpe subword_nmt --remove-bpe --save-scores mos_scores/mos5_finetune.npy
