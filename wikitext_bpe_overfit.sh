## eval
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_last.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe

# store overfit
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_last.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/wikitext103-bpe/dstore_last --knn-keytype 'last_ffn_input' \
    --dstore-size 153225485 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16

# build index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103-bpe/dstore_last \
    --dstore_size 153225485 \
    --faiss_index checkpoints/wikitext103-bpe/knn_last.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore-fp16 --dimension 1024

# eval with index
# no recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_last.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_last \
    --indexfile checkpoints/wikitext103-bpe/knn_last.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_ffn_input \
    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe


# recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe/checkpoint_last.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_last \
    --indexfile checkpoints/wikitext103-bpe/knn_last.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe


# continue training till overfit
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-overfit \
  --arch transformer_lm_wikibpe \
  --restore-file checkpoints/wikitext103-bpe/checkpoint_last.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --max-update 28600 --optimizer nag --lr 1e-3 --clip-norm 0.1 \
  --max-tokens 3072 --update-freq 6 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-overfit/checkpoint242.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe


# store continue training overfit
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-overfit/checkpoint242.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/wikitext103-bpe/dstore_242 --knn-keytype 'last_ffn_input' \
    --dstore-size 153225485 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16

# build index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103-bpe/dstore_242 \
    --dstore_size 153225485 \
    --faiss_index checkpoints/wikitext103-bpe/knn_242.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore-fp16 --dimension 1024

# eval with index
# no recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-overfit/checkpoint242.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_242 \
    --indexfile checkpoints/wikitext103-bpe/knn_242.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_ffn_input \
    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe


# recompute
python eval_lm.py data-bin/wikitext103-bpe \
    --path checkpoints/wikitext103-bpe-overfit/checkpoint242.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_242 \
    --indexfile checkpoints/wikitext103-bpe/knn_242.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 153225485 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe


### smaller training data, 1%
TEXT=examples/language_model/wikitext-103
python preprocess.py \
    --only-source \
    --srcdict data-bin/wikitext103-bpe/dict.txt \
    --trainpref $TEXT/wiki.train.small.tokens.bpe \
    --validpref $TEXT/wiki.valid.tokens.bpe \
    --testpref $TEXT/wiki.test.tokens.bpe \
    --destdir data-bin/wikitext103-bpe-small \
    --workers 20

python eval_lm.py data-bin/wikitext103-bpe-small \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe

# create datastore with best
python eval_lm.py data-bin/wikitext103-bpe-small \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/wikitext103-bpe/dstore_small --knn-keytype 'last_ffn_input' \
    --dstore-size 1113601 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16

# build index
python build_dstore.py \
    --dstore_mmap checkpoints/wikitext103-bpe/dstore_small \
    --dstore_size 1113601 \
    --faiss_index checkpoints/wikitext103-bpe/knn_small.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore-fp16 --dimension 1024

# eval with index
# no recompute
python eval_lm.py data-bin/wikitext103-bpe-small \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_small \
    --indexfile checkpoints/wikitext103-bpe/knn_small.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 1113601 --knn-keytype last_ffn_input \
    --knn-sim-func "do_not_recomp_l2" --no-load-keys \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe


# recompute
python eval_lm.py data-bin/wikitext103-bpe-small \
    --path checkpoints/wikitext103-bpe/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/wikitext103-bpe/dstore_small \
    --indexfile checkpoints/wikitext103-bpe/knn_small.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 1113601 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 --bpe subword_nmt --remove-bpe


# continue training till overfit
CUDA_VISIBLE_DEVICES=2,3 python train.py --task language_modeling \
    data-bin/wikitext103-bpe-small \
  --save-dir checkpoints/wikitext103-bpe-overfit \
  --arch transformer_lm_wikibpe \
  --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters \
  --max-update 28600 --optimizer nag --lr 1e-4 --clip-norm 0.1 \
  --max-tokens 3072 --update-freq 1 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16

# eval overfit model
python eval_lm.py data-bin/wikitext103-bpe-small \
    --path checkpoints/wikitext103-bpe-overfit/checkpoint10.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --bpe subword_nmt --remove-bpe
