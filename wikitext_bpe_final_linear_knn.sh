# store non-tied
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/wikitext103-bpe/last_linear_inp/dstore --knn-keytype 'last_linear_input' \
    --dstore-size 153225485 --model-overrides "{'knn_keytype': 'last_linear_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16

# build index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103-bpe/last_linear_inp/dstore \
    --dstore_size 153225485 \
    --faiss_index checkpoints/wikitext103-bpe/last_linear_inp/knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore-fp16 --dimension 1024

# eval with index
# no recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/last_linear_inp/dstore \
    --indexfile checkpoints/wikitext103-bpe/last_linear_inp/knn.index  \
    --model-overrides "{'knn_keytype': 'last_linear_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_linear_input \
    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe

# recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/last_linear_inp/dstore \
    --indexfile checkpoints/wikitext103-bpe/last_linear_inp/knn.index  \
    --model-overrides "{'knn_keytype': 'last_linear_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_linear_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe

## USE IP metric
# build index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103-bpe/last_linear_inp/dstore \
    --dstore_size 153225485 \
    --faiss_index checkpoints/wikitext103-bpe/last_linear_inp/knn_ip.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore-fp16 --dimension 1024 --metric ip

# no recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/last_linear_inp/dstore \
    --indexfile checkpoints/wikitext103-bpe/last_linear_inp/knn.index --faiss-metric-type ip \
    --model-overrides "{'knn_keytype': 'last_linear_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_linear_input \
    --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe

# recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/last_linear_inp/dstore \
    --indexfile checkpoints/wikitext103-bpe/last_linear_inp/knn.index --faiss-metric-type ip \
    --model-overrides "{'knn_keytype': 'last_linear_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_linear_input \
    --knn-sim-func "dot" \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe
