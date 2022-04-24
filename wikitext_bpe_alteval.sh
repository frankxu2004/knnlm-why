# LM 3072, context-window 2560, none

## eval train
python eval_lm.py data-bin/wikitext103-bpe  \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt  \
    --sample-break-mode none --max-tokens 3072 --softmax-batch 1024 \
    --gen-subset train --bpe subword_nmt --remove-bpe \
    --save-tokens overfit_analysis/train_tokens.npy --save-scores overfit_analysis/train_lm_scores.npy

## eval valid
python eval_lm.py data-bin/wikitext103-bpe  \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt  \
    --sample-break-mode none --max-tokens 3072 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --save-tokens overfit_analysis/tokens.npy --save-scores overfit_analysis/lm_scores.npy

## eval with KNN
python eval_lm.py data-bin/wikitext103-bpe  \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt  \
    --sample-break-mode none --max-tokens 3072 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore \
    --indexfile checkpoints/wikitext103-bpe/knn_prune.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_ffn_input \
    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe \
    --save-knn-scores overfit_analysis/knn_scores.npy

## eval with KNN recomp
python eval_lm.py data-bin/wikitext103-bpe  \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt  \
    --sample-break-mode none --max-tokens 3072 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore \
    --indexfile checkpoints/wikitext103-bpe/knn_prune.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe \
    --save-knn-scores overfit_analysis/knn_recomp_scores.npy

## eval with KNN ip
python eval_lm.py data-bin/wikitext103-bpe  \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt  \
    --sample-break-mode none --max-tokens 3072 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore \
    --indexfile checkpoints/wikitext103-bpe/knn_ip.index --faiss-metric-type ip \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_ffn_input \
    --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe \
    --save-knn-scores overfit_analysis/knn_ip_scores.npy

## eval with KNN ip recomp
python eval_lm.py data-bin/wikitext103-bpe  \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt  \
    --sample-break-mode none --max-tokens 3072 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore \
    --indexfile checkpoints/wikitext103-bpe/knn_ip.index --faiss-metric-type ip \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_ffn_input \
    --knn-sim-func "dot" \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe \
    --save-knn-scores overfit_analysis/knn_ip_recomp_scores.npy


# overfitted model
## eval train
python eval_lm.py data-bin/wikitext103-bpe  \
    --path checkpoints/wikitext103-bpe-overfit-new/checkpoint129.pt  \
    --sample-break-mode none --max-tokens 3072 --softmax-batch 1024 \
    --gen-subset train --bpe subword_nmt --remove-bpe \
    --save-scores overfit_analysis/train_overfit129_lm_scores.npy

## eval valid
python eval_lm.py data-bin/wikitext103-bpe  \
    --path checkpoints/wikitext103-bpe-overfit-new/checkpoint129.pt  \
    --sample-break-mode none --max-tokens 3072 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe \
    --save-scores overfit_analysis/overfit129_lm_scores.npy
