# non tied
python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-kv2 \
  --arch transformer_lm_wikibpe \
  --max-update 386000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
  --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
  --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16 \
  --pseudo-vocab-ratio 2 --criterion agg_softmax

## eval
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-kv2/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 --pseudo-vocab-ratio 2 \
    --gen-subset valid --bpe subword_nmt --remove-bpe


## eval tied
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-tied/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe


## store tied train
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-tied/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/wikitext103-bpe-tied/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 153225485 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16

# build index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103-bpe-tied/dstore \
    --dstore_size 153225485 \
    --faiss_index checkpoints/wikitext103-bpe-tied/knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore-fp16 --dimension 1024

# eval with index
# no recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-tied/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe-tied/dstore \
    --indexfile checkpoints/wikitext103-bpe-tied/knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_ffn_input \
    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe

# recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-tied/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe-tied/dstore \
    --indexfile checkpoints/wikitext103-bpe-tied/knn.index \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe


# store non-tied
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/wikitext103-bpe/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 153225485 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16

# build index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103-bpe/dstore \
    --dstore_size 153225485 \
    --faiss_index checkpoints/wikitext103-bpe/knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore-fp16 --dimension 1024

# eval with index
# no recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore \
    --indexfile checkpoints/wikitext103-bpe/knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_ffn_input \
    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe


# recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore \
    --indexfile checkpoints/wikitext103-bpe/knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe
