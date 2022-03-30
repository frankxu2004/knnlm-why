for i in {1..233}
do
  python eval_lm.py data-bin/wikitext103-bpe \
      --path checkpoints/wikitext103-bpe-overfit/checkpoint${i}.pt \
      --sample-break-mode complete --max-tokens 3072 \
      --context-window 2560 --softmax-batch 1024 \
      --gen-subset valid --bpe subword_nmt --remove-bpe
done