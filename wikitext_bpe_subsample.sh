## 0.05
#python analysis/subsample_datastore_0.05.py
#
## build index
#python build_dstore.py \
#    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.05 \
#    --dstore_size 7661274 \
#    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.05.index \
#    --num_keys_to_add_at_a_time 500000 \
#    --starting_point 0 --dstore-fp16 --dimension 1024
#
## eval with index
## no recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.05 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.05.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 7661274 --knn-keytype last_ffn_input \
#    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.05_knn_scores.npy
#
## recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.05 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.05.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 7661274 --knn-keytype last_ffn_input \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.05_knn_recomp_scores.npy
#
## 0.1
#python analysis/subsample_datastore_0.1.py
#
## build index
#python build_dstore.py \
#    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.1 \
#    --dstore_size 15322548 \
#    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.1.index \
#    --num_keys_to_add_at_a_time 500000 \
#    --starting_point 0 --dstore-fp16 --dimension 1024
#
## eval with index
## no recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.1 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.1.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 15322548 --knn-keytype last_ffn_input \
#    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.1_knn_scores.npy
#
## recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.1 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.1.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 15322548 --knn-keytype last_ffn_input \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.1_knn_recomp_scores.npy
#
## 0.2
#python analysis/subsample_datastore_0.2.py
#
## build index
#python build_dstore.py \
#    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.2 \
#    --dstore_size 30645096 \
#    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.2.index \
#    --num_keys_to_add_at_a_time 500000 \
#    --starting_point 0 --dstore-fp16 --dimension 1024
#
## eval with index
## no recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.2 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.2.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 30645096 --knn-keytype last_ffn_input \
#    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.2_knn_scores.npy
#
## recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.2 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.2.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 30645096 --knn-keytype last_ffn_input \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.2_knn_recomp_scores.npy
#
#
## 0.3
#python analysis/subsample_datastore_0.3.py
#
## build index
#python build_dstore.py \
#    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.3 \
#    --dstore_size 45967645 \
#    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.3.index \
#    --num_keys_to_add_at_a_time 500000 \
#    --starting_point 0 --dstore-fp16 --dimension 1024
#
## eval with index
## no recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.3 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.3.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 45967645 --knn-keytype last_ffn_input \
#    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.3_knn_scores.npy
#
## recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.3 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.3.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 45967645 --knn-keytype last_ffn_input \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.3_knn_recomp_scores.npy
#
#
## 0.4
#python analysis/subsample_datastore_0.4.py
#
## build index
#python build_dstore.py \
#    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.4 \
#    --dstore_size 61290194 \
#    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.4.index \
#    --num_keys_to_add_at_a_time 500000 \
#    --starting_point 0 --dstore-fp16 --dimension 1024
#
## eval with index
## no recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.4 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.4.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 61290194 --knn-keytype last_ffn_input \
#    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.4_knn_scores.npy
#
## recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.4 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.4.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 61290194 --knn-keytype last_ffn_input \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.4_knn_recomp_scores.npy
#
## 0.5
#python analysis/subsample_datastore_0.5.py
#
## build index
#python build_dstore.py \
#    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.5 \
#    --dstore_size 76612742 \
#    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.5.index \
#    --num_keys_to_add_at_a_time 500000 \
#    --starting_point 0 --dstore-fp16 --dimension 1024
#
## eval with index
## no recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.5 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.5.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 76612742 --knn-keytype last_ffn_input \
#    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.5_knn_scores.npy
#
## recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.5 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.5.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 76612742 --knn-keytype last_ffn_input \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.5_knn_recomp_scores.npy
#
#
## 0.6
#python analysis/subsample_datastore_0.6.py
#
## build index
#python build_dstore.py \
#    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.6 \
#    --dstore_size 91935291 \
#    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.6.index \
#    --num_keys_to_add_at_a_time 500000 \
#    --starting_point 0 --dstore-fp16 --dimension 1024
#
## eval with index
## no recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.6 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.6.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 91935291 --knn-keytype last_ffn_input \
#    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.6_knn_scores.npy
#
## recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.6 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.6.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 91935291 --knn-keytype last_ffn_input \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.6_knn_recomp_scores.npy


## 0.7
#python analysis/subsample_datastore_0.7.py
#
## build index
#python build_dstore.py \
#    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.7 \
#    --dstore_size 107257840 \
#    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.7.index \
#    --num_keys_to_add_at_a_time 500000 \
#    --starting_point 0 --dstore-fp16 --dimension 1024
#
## eval with index
## no recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.7 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.7.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 107257840 --knn-keytype last_ffn_input \
#    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.7_knn_scores.npy
#
## recompute
#python eval_lm.py data-bin/wikitext103-bpe \
#    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
#    --sample-break-mode complete --max-tokens 3072 \
#    --context-window 2560 --softmax-batch 1024 \
#    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.7 \
#    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.7.index  \
#    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
#    --k 1024 --lmbda 0.25 --dstore-size 107257840 --knn-keytype last_ffn_input \
#    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.7_knn_recomp_scores.npy


# 0.8
python analysis/subsample_datastore_0.8.py

# build index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.8 \
    --dstore_size 122580388 \
    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.8.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore-fp16 --dimension 1024

# eval with index
# no recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.8 \
    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.8.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 122580388 --knn-keytype last_ffn_input \
    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.8_knn_scores.npy

# recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.8 \
    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.8.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 122580388 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.8_knn_recomp_scores.npy

# 0.9
python analysis/subsample_datastore_0.9.py

# build index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103-bpe/dstore_subsampled_0.9 \
    --dstore_size 137902936 \
    --faiss_index checkpoints/wikitext103-bpe/knn_subsampled_0.9.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore-fp16 --dimension 1024

# eval with index
# no recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.9 \
    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.9.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 137902936 --knn-keytype last_ffn_input \
    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.9_knn_scores.npy

# recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_subsampled_0.9 \
    --indexfile checkpoints/wikitext103-bpe/knn_subsampled_0.9.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 137902936 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe --save-knn-scores 0.9_knn_recomp_scores.npy
