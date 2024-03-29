# SIR-ABSC

## We are updating minor modifications. The final code will be available by 3.31.2024.

## Introduction
We present a simple, but effective method to incorporate syntactic  dependency information  directly into transformer-based language models (e.g. RoBERTa) for tasks such as Aspect-Based Sentiment Classification (ABSC), where the desired  output depends on specific input tokens. In contrast to prior approaches to ABSC that capture syntax by combining language models with graph neural networks over dependency trees, our model, Syntax-Integrated RoBERTa for ABSC (SIR-ABSC) incorporates syntax directly into the language model by using a novel aggregator token. SIR-ABSC outperforms these more complex models, yielding new  state-of-the-art results on ABSC. 

<p align="center">
  <img src="Overall.png" />
</p>
  
## Requirements
Install required dependencies via
```
pip install -r requirements.txt
```

## Reproducing the results:
To reproduce the results please follow the instructions below.
                                                                                      
  1. Download and convert BERT-related files & checkpoints:
     ```
     unzip uncased_L-12_H-768_A-12.zip
     BERT_BASE_DIR='./uncased_L-12_H-768_A-12'
     python3 convert_tf_to_pytorch/convert_tf_checkpoint_to_pytorch.py \
     --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
     --bert_config_file $BERT_BASE_DIR/bert_config.json \
     --pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin
     ```
  2. Create dependency graphs by running following codes:
     ```
     python generateGraph_spacy.py --parser_info 'spacy_sm_3.3.0' --create_global_vocabs True
     python data_utils_bert.py --parser_info 'spacy_sm_3.3.0'
     python data_utils_roberta.py --parser_info 'spacy_sm_3.3.0'
     ```
  3. Run the following script to train a model:
     ```
     python run_classifier_word_roberta.py \
     --task_name=restaurant \
     --model_name roberta_gcls \
     --graph_type 'dg' \
     --input_format 'gX' \
     --constant_vdc '0,0,0,1,1,1,2,2,2,3,3,3' \
     --VDC_threshold 0.8 \
     --use_hVDC True \
     --g_pooler 's_g_new_2' \
     --use_DEP True \
     --DAA_start_layer -1 \
     --do_auto False \
     --parser_info 'spacy_sm_3.3.0' \
     --learning_rate 2.0e-5 \
     --num_train_epochs 30.0 \
     --eval_start_epoch 5
     ```

## Citation
If you use our work, please cite:
```
@article{sirabsc2023,
  title={SIR-ABSC: Incorporating Syntax into RoBERTa-based Sentiment Analysis Models with a Special Aggregator Token},
  author={Cho, Ikhyun and Jung, Yoonhwa and Hockenmaier, Julia},
  journal={EMNLP Findings},
  year={2023},
}
```
## Acknowledgement

The implementation of SIR-ABSC relies on resources from [ASGCN](https://github.com/GeneZC/ASGCN). We thank the original authors for their open-sourcing.
