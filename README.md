# Paraphrase-Generation
Work in progress

## Overview
The Transformer from “Attention is All You Need” is currently the dominant architecture in NLP. It provides a new architecture for many NLP related tasks. 
After searching the web for a bit, I have seen that this model is  commonly used for translating languages. For this project, I wanted to train the transformer model on a paraphrasing dataset. 

```preprocess.py```: It tokenizes each sentence from the PAWS dataset, creates the vocabulary, converts each word to indexes, and saves them to a json and h5 file

```dataloader.py```: It reads from the .json and .h5 files that have been created previously, and adds padding. It inherits from the torch.utils.data Dataset class and will be used as the dataloader when training.

```model.py```: Contains the code for the Transformer model

```train.py```: Trains the model

```paraphrase.py```: Generate a paraphrase from trained model

## Getting Started

```
Install requirements
pip3 install torch==1.12.0
pip3 install torchtext==0.6.0
pip3 install tqdm==4.64.0
pip3 install h5py==3.7.0
pip3 install nltk==3.7

How to use
1. python preprocess.py
2. python train.py
3. python paraphrase.py
```

## To Do List
- Add validation
- Add argument parser
- Add a config file where filenames will be stored 
- Train model with other paraphrasing datasets 

## Acknowledgement
- https://github.com/bentrevett/pytorch-seq2seq
- https://arxiv.org/abs/1706.03762
- https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

## Contact
- https://github.com/miztqq
