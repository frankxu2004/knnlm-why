## preprocess
TEXT=examples/language_model/wikitext-103
python preprocess.py \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens.bpe \
    --validpref $TEXT/wiki.valid.tokens.bpe \
    --testpref $TEXT/wiki.test.tokens.bpe \
    --destdir data-bin/wikitext103-bpe \
    --workers 20

python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe \
  --arch transformer_lm_big --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 16000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 3072 --sample-break-mode none \
  --max-tokens 3072 --update-freq 3 \
  --fp16 \
  --max-update 286000 --ddp-backend=no_c10d

## eval
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe

## store train
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/wikitext103-bpe/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 153225485 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16
