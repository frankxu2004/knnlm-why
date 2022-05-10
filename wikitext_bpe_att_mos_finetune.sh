# ATT MOS k=3
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-mos3-att-finetune \
  --arch transformer_lm_wikibpe  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --finetune-mos --k-mos 3 --knn-keytype last_ffn_input --use-last-ffn-input \
  --max-update 286000 --optimizer nag --lr 5e-2 --clip-norm 100 \
  --max-tokens 9216 --update-freq 1 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16


# ATT MOS k=3 + finetune output embedding
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-mos3-att-embed-finetune \
  --arch transformer_lm_wikibpe  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --finetune-mos --finetune-out-embed --k-mos 3 --knn-keytype last_ffn_input --use-last-ffn-input \
  --max-update 286000 --optimizer nag --lr 5e-2 --clip-norm 100 \
  --max-tokens 9216 --update-freq 1 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

# ATT MOS k=2 + finetune output embedding
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-mos2-att-embed-finetune \
  --arch transformer_lm_wikibpe  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --finetune-mos --finetune-out-embed --k-mos 2 --knn-keytype last_ffn_input --use-last-ffn-input \
  --max-update 286000 --optimizer nag --lr 1e-3 --clip-norm 100 \
  --max-tokens 9216 --update-freq 1 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

## eval
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-mos3-att-finetune/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --k-mos 3 \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --save-scores mos_scores/mos3_att_finetune.npy

python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-mos2-att-finetune/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --k-mos 2 \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --save-scores mos_scores/mos2_att_finetune.npy

python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-mos3-att-embed-finetune/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --k-mos 3 \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --save-scores mos_scores/mos3_att_embed_finetune.npy

python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-mos2-att-embed-finetune/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --k-mos 2 \
    --model-overrides "{'knn_keytype': 'last_ffn_input', 'use_last_ffn_input': True}" \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --save-scores mos_scores/mos2_att_embed_finetune.npy
