# Evaluation of large language models for the generation of knowledge graphs

This repository contains:

* A technical report: [PDF](COST_DKG_report.pdf)
* Code to download graphs from DBpedia in rdf and csv formats: [query.py](download_graph.py)
* The required code to fine-tune an instance of Mistral on the downloaded graph: [finetune.py](finetune.py).
* A script to run inference on the fine-tuned model: [inference.py](inference.py)
* The code to train and evaluate tokenizers: [tokenizer.py](tokenizer.py)