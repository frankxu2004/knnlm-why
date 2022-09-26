# 0.05
python analysis/subsample_datastore_0.05.py

# build index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.05 \
    --dstore_size 7661274 \
    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.05.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore-fp16 --dimension 1024

# eval with index
# no recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.05 \
    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.05.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 7661274 --knn-keytype last_ffn_input \
    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.05_knn_scores_2.npy \
    --save-queries all_queries.npy


# 0.1
python analysis/subsample_datastore_0.1.py

# build index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.1 \
    --dstore_size 15322548 \
    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.1.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore-fp16 --dimension 1024

# eval with index
# no recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.1 \
    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.1.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 15322548 --knn-keytype last_ffn_input \
    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.1_knn_scores_2.npy \
    --save-queries all_queries_1.npy


# 0.2
python analysis/subsample_datastore_0.2.py

# build index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.2 \
    --dstore_size 30645096 \
    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.2.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore-fp16 --dimension 1024

# eval with index
# no recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.2 \
    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.2.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 30645096 --knn-keytype last_ffn_input \
    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.2_knn_scores_2.npy \
    --save-queries all_queries_2.npy


