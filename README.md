# Why do Nearest Neighbor Language Models Work?

Language models (LMs) compute the probability of a text by sequentially computing a representation of an already-seen context and using this representation to predict the next word. Currently, most LMs calculate these representations through a neural network consuming the immediate previous context. However recently, retrieval-augmented LMs have shown to improve over standard neural LMs, by accessing information retrieved from a large datastore, in addition to their standard, parametric, next-word prediction. In this paper, we set out to understand why retrieval-augmented language models, and specifically why k-nearest neighbor language models (kNN-LMs) perform better than standard parametric LMs, even when the k-nearest neighbor component retrieves examples from the same training set that the LM was originally trained on. To this end, we perform a careful analysis of the various dimensions over which kNN-LM diverges from standard LMs, and investigate these dimensions one by one. Empirically, we identify three main reasons why kNN-LM performs better than standard LMs: using a different input representation for predicting the next tokens, approximate kNN search, and the importance of softmax temperature for the kNN distribution. Further, we incorporate these insights into the model architecture or the training procedure of the standard parametric LM, improving its results without the need for an explicit retrieval component.

This code pertains to the paper: [Why do Nearest Neighbor Language Models Work?](https://arxiv.org/abs/2301.02828). 
This repository is a fork of the [original knnlm](https://github.com/urvashik/knnlm) repository. Kudos to the authors who made this research possible.

### Environment

* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library with the `--cuda_ext` and `--deprecated_fused_adam` options

Before starting, make sure you install this forked version of Fairseq (after pulling the code, from the project directory) and [FAISS](https://github.com/facebookresearch/faiss/wiki), preferably the FAISS GPU version:
```bash
pip install --editable .

pip install faiss
```


### Download Large Files
* Pretrained models
* Pre-processed data 
* Populated datastore and trained FAISS indexes
* TODO: update with the download links


### Running the experiments in the paper

More details comming soon, but most experiments based on Wikitext103 are organized into scripts that are self-explanatory in the comments in `wikitext_bpe*.sh` files in the root directory.

### A Note about Hardware

Experiments for this paper were conducted on machines that contain 500GB of RAM, NVIDIA RTX 8000 48GB GPUs and flash storage (SSDs). 
Saving the Wikitext-103 datastore, even in fp16 as we did requires 300GB of disk space. The speed of saving the datastore, building the FAISS index and evaluating the nearest neighbors language model heavily depends on the amount of RAM available for each job. Some of these steps can be sped up by parallelizing, which we leave for users to do in order to best cater to their setup.

If you are working with a remote cluster, please note that we use [memmaps](https://numpy.org/doc/1.18/reference/generated/numpy.memmap.html) for saving the datastore. This allows us to keep the data on disk while accessing it by loading small chunks into memory, depending on the available RAM. This means there are a large number of disk seeks. In order to prevent slowing down your entire cluster, we suggest always reading/writing this data to/from local disks (as opposed to NFS directories), and flash storage is best for faster access.
