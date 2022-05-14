from fairseq import checkpoint_utils
import numpy as np

state = checkpoint_utils.load_checkpoint_to_cpu('checkpoints/wikitext103-bpe/checkpoint_best.pt')

wemb = state["model"]['decoder.embed_out'].numpy()
np.savez('output_embedding.npz', decoder_Wemb=wemb)
