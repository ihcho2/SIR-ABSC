#!/bin/bash

for i in $(seq 1 5)
do
echo $i;
python run_classifier_word.py \
--task_name=laptop \
--data_dir=/home/ikhyuncho23/sentiment/aspect_sentiment/datasets/semeval14/laptops/3way \
--vocab_file=/home/ikhyuncho23/sentiment/bert/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=/home/ikhyuncho23/sentiment/bert/uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=/home/ikhyuncho23/sentiment/bert/uncased_L-12_H-768_A-12/pytorch_model.bin \
--max_seq_length=128 \
--train_batch_size 32 \
--eval_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 6.0 \
--model_name td_bert \
--local_rank 0 \
--gpu_id 0 \
--output_dir log/lap_td_bert_3way_10 \
--para_LSR 0.2
done
python std.py
