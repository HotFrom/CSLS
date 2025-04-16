# CSLS
This repository contains the implementation of our Internetware 2025 paper:  > **Line-level Semantic Structure Learning for Code Vulnerability Detection**  
# Line-level Semantic Structure Learning for Code Vulnerability Detection

This repository contains the implementation of our Internetware 2025 paper:

> **Line-level Semantic Structure Learning for Code Vulnerability Detection**  
> _Ziliang Wang, et al._  
> Accepted at Internetware 2025  
> Dataset: [Devign](https://github.com/microsoft/Devign)

## 📌 Introduction

This project presents a novel vulnerability detection model that leverages both **line-level semantics** and **global structural context** in source code.  
We integrate sentence-level representations through a Transformer-based encoder, combined with a global [CLS] representation, to enhance vulnerability prediction accuracy.

## 📁 Dataset

We use the **Devign** dataset for all experiments. The dataset is divided into:
- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`


2. Training, Evaluation, and Testing
Run the following command:

bash
复制
编辑
CUDA_VISIBLE_DEVICES=1,2,0 python run.py \
  --output_dir=./saved_models \
  --epoch 4 \
  --block_size 1024 \
  --train_batch_size 36 \
  --eval_batch_size 36 \
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 1 \
  --model_type=roberta \
  --tokenizer_name=microsoft/unixcoder-nine \
  --model_name_or_path=microsoft/unixcoder-nine \
  --do_train \
  --do_eval \
  --do_test \
  --train_data_file=../dataset/data1/train.jsonl \
  --eval_data_file=../dataset/data1/valid.jsonl \
  --test_data_file=../dataset/data1/test.jsonl \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 123456
🧪 Experimental Results

Metric	Value
Accuracy	70.17%
Precision	71.24%
Recall	58.80%
F1 Score	64.43%

Citation
If you find our work useful, please cite:

bibtex

@inproceedings{wang2025line,
  title={Line-level Semantic Structure Learning for Code Vulnerability Detection},
  author={Wang, Ziliang and et al.},
  booktitle={Proceedings of the Internetware 2025},
  year={2025}
}
