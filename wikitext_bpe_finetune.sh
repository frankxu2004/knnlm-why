# finetune load centroid
CUDA_VISIBLE_DEVICES=1 python train.py --task language_modeling \
    data-bin/wikitext103-bpe \
  --save-dir checkpoints/wikitext103-bpe-centroid-finetune \
  --arch transformer_lm_wikibpe --restore-file checkpoints/wikitext103-bpe/checkpoint_best.pt \
  --finetune-centroids --reset-optimizer --reset-dataloader --reset-meters \
  --criterion agg_softmax --use-last-ffn-input --knn-keytype last_ffn_input \
  --load-centroids checkpoints/wikitext103-bpe/centroids.npy  \
  --load-centroid-distribution checkpoints/wikitext103-bpe/cluster_freq.npz \
  --max-update 28600 --max-lr 1.0 --t-mult 2 --lr-period-updates 27000 --lr-scheduler cosine --lr-shrink 0.75 \
  --warmup-updates 1600 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
  --max-tokens 3072 --update-freq 24 --tokens-per-sample 3072 --seed 1 \
  --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --fp16
