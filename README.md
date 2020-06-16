# CamemBERT: a Tasty French Language Model

## Introduction

[CamemBERT](https://arxiv.org/abs/1911.03894) is a state-of-the-art language model for French based on the RoBERTa model.

It is now available on Hugging Face in 6 different versions with varying number of parameters, amount of pretraining data and pretraining data source domains.

For further information

## Setting up camembert_finetuned library 


`git clone https://github.com/benjamin-mlr/camembert_finetune.git`

`cd ./camembert_finetune`

`sh ./install.sh`

`source activate camembert_finetune` 


## Tuned Models

This project was partially built using a fork of the [transformers](https://huggingface.co/transformers) library 

|Task |  Pretrained Model                          | #params                        | Arch. | Pre-Training data                    |   Fine-Tuning Data | 
|--------|--------------------------------|--------------------------------|-------|-----------------------------------|-----------------------------| 
| Part-of-speech Tagging | `camembert-base` | 110M+   | Base  | CCNET (130+ GB of text)            | French Partut |
| Name Entity Recognition | `camembert-base` | 110M+   | Base  | OSCAR (138 GB of text)            | French Treebank | 

### Download

`mkdir ./camembert_finetune/checkpoints`    

#### POS

`cd ./camembert_finetune/checkpoints`   
`wget dl.fbaipublicfiles.com/camembert/camembert_finetuned_pos.tar.gz`     
`tar -xzvf camembert_finetuned_pos.tar.gz`

#### NER 

`cd ./camembert_finetune/checkpoints`  
`wget dl.fbaipublicfiles.com/fairseq/models/camembert_finetuned_ner.tar.gz`      
`tar -xzvf camembert_finetuned_ner.tar.gz`
 

## Using Tuned models 

### Interact with Camembert 

`python ./camembert_predict.py  --task pos`


`python ./camembert_predict.py  --task ner`

### Prediction 

`python ./camembert_predict.py --task pos --input_file ./input.conllu --output_file ./output_pos.conllu` 


`python ./camembert_predict.py --task ner --input_file ./input.conllu --output_file ./output_ner.conllu`
 

### Evaluation  

For evaluation, provide the gold file along with the `input_file`

`python ./camembert_predict.py --task ner --input_file ./input.conllu --output_file ./output_ner.conllu --gold_file ./gold_ner.conllu `

To have more granular scores set `--score_details 1` 


NB : camembert supports CoNLL-U file: 
- Word lines containing the annotation of a word/token in 10 fields separated by single tab characters; see below.
- Blank lines marking sentence boundaries.
- Comment lines starting with hash (#)

more details https://universaldependencies.org/format.html

 