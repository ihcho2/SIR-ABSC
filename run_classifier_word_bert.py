# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import csv
import os
import logging
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from modeling import BertConfig, BertForSequenceClassification, BertForSequenceClassification_gcls, BertForSequenceClassification_TD
from modeling_m1 import BertForSequenceClassification_gcls_m1
from optimization import BERTAdam

from configs import get_config
from models import CNN, CLSTM, PF_CNN, TCN, Bert_PF, BBFC, TC_CNN, RAM, IAN, ATAE_LSTM, AOA, MemNet, Cabasc, TNet_LF, MGAN, BERT_IAN, TC_SWEM, MLP, AEN_BERT, TD_BERT, TD_BERT_QA, DTD_BERT, TD_BERT_with_GCN, BERT_FC_GCN
from utils.data_util_bert import ReadData, RestaurantProcessor, LaptopProcessor, TweetProcessor, MamsProcessor
from utils.save_and_load import load_model_m1
import torch.nn.functional as F
from sklearn.metrics import f1_score

from data_utils import *
from transformers import BertTokenizer
from torch.distributions.bernoulli import Bernoulli

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    # print(outputs)
    return np.sum(outputs == labels)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan


class Instructor:
    def __init__(self, args):
        self.opt = args
        #self.writer = SummaryWriter(log_dir=self.opt.output_dir)  # tensorboard
        bert_config = BertConfig.from_json_file(args.bert_config_file)

        if args.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                    args.max_seq_length, bert_config.max_position_embeddings))

        # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        # os.makedirs(args.output_dir, exist_ok=True)
        
        self.dataset = ReadData(self.opt)  # Read the data and preprocess it
        self.num_train_steps = None
        self.num_train_steps = int(len(
            self.dataset.train_examples) / self.opt.train_batch_size / self.opt.gradient_accumulation_steps * self.opt.num_train_epochs)
        self.opt.label_size = len(self.dataset.label_list)
        args.output_dim = len(self.dataset.label_list)
        
        self.train_tran_indices = self.dataset.train_tran_indices
        self.train_span_indices = self.dataset.train_span_indices
        self.eval_tran_indices = self.dataset.eval_tran_indices
        self.eval_span_indices = self.dataset.eval_span_indices
        
        if args.model_name in ['bert_gcls', 'bert_gcls_m1', 'bert_td']:
            self.train_extended_attention_mask = self.dataset.train_extended_attention_mask.to(args.device)
            self.eval_extended_attention_mask = self.dataset.eval_extended_attention_mask.to(args.device)
            
            self.train_VDC_info = self.dataset.train_VDC_info.to(args.device)
            self.eval_VDC_info = self.dataset.eval_VDC_info.to(args.device)
            
            self.train_AAW_mask_token = self.dataset.train_AAW_mask_token.to(args.device)
            self.eval_AAW_mask_token = self.dataset.eval_AAW_mask_token.to(args.device)
            
            self.train_current_VDC = self.dataset.train_current_VDC.to(args.device)
            self.eval_current_VDC = self.dataset.eval_current_VDC.to(args.device)
            
#             self.train_VDC_weight = self.dataset.train_VDC_weight.to(args.device)
#             self.eval_VDC_weight = self.dataset.eval_VDC_weight.to(args.device)
            
            if task_name == 'mams':
                self.validation_extended_attention_mask = self.dataset.validation_extended_attention_mask.to(args.device)
                self.validation_VDC_info = self.dataset.validation_VDC_info.to(args.device)
                self.validation_AAW_mask_token = self.dataset.validation_AAW_mask_token.to(args.device)
                self.validation_current_VDC = self.dataset.validation_current_VDC.to(args.device)
#                 self.validation_VDC_weight = self.dataset.validation_VDC_weight.to(args.device)
                
        print("label size: {}".format(args.output_dim))

        # 初始化模型
        print("initialize model ...")
        
        os.makedirs(args.model_save_path, exist_ok=True)
        
        if args.model_class == BertForSequenceClassification:
            self.model = BertForSequenceClassification(bert_config, len(self.dataset.label_list))
#         if args.model_class == BertForSequenceClassification_TD:
#             self.model = BertForSequenceClassification_TD(bert_config, len(self.dataset.label_list))
        elif args.model_class == BertForSequenceClassification_gcls:
            self.model = BertForSequenceClassification_gcls(bert_config, len(self.dataset.label_list), g_pooler=args.g_pooler,
                                                            plus_layer = args.plus_layer, minus_layer = args.minus_layer)
        elif args.model_class == BertForSequenceClassification_gcls_m1:
            self.model = BertForSequenceClassification_gcls_m1(bert_config, len(self.dataset.label_list),g_pooler = args.g_pooler,
                                                                                        automation_type = args.automation_type,
                                                                                        auto_k = args.auto_k, 
                                                                                        embed_dense = args.embed_dense)
        else:
            self.model = model_classes[args.model_name](bert_config, args)
        
        if self.opt.model_name in ['bert_gcls_m1', 'bert_td'] or self.opt.g_pooler in ['s_g_concat', 's_g_att_ec', 'dgedt_1', 'dgedt_2',
                                                                            'dgedt_3', 's_g_att_ec_2', 's_g_avg_var_1',
                                                                            's_g_t_avg_avg_var_1', 's_g_t_max_avg_var_1',
                                                                            's_g_t_avg_att_ec', 's_g_t_max_att_ec',
                                                                            's_g_att', 's_g_avg', 's_g_g2_t_avg_avg_var_1',
                                                                            's_g_g2_t_avg_t2_avg_avg_var_1', 's_g_g2',
                                                                            's_g2_t_avg', 's_g2_t2_avg', 's_g2_avg', 'sggcn_1',
                                                                            'sggcn_2', 's_g_inter_t_avg_avg_var1', 't_avg',
                                                                            'g_t_avg_avg_var_1', 'g_t_avg_att', 'g_t_avg_ec',
                                                                            's_g_t_avg_ec', 'g', 's', 's_g_t_avg_avg_var_2',
                                                                            's_g_t_avg_concat', 's_g_sg_avg',
                                                                            's_g_t_avg_avg_var_3', 's_g_sg_ec', 's_g_max',
                                                                            's_g_max_avg']:
            
            self.model = load_model_m1(self.model, self.opt.init_checkpoint)
            
            # 검토한번해보셈: self.model.bert.___.weight.data => 전부 0은 아니겟지?
        else:
            if 'pytorch_model.bin' in self.opt.init_checkpoint:
                self.model.bert.load_state_dict(torch.load(self.opt.init_checkpoint, map_location='cpu'))
                print('-'*77)
                print('Loading from ', self.opt.init_checkpoint)
                print('-'*77)
        
        self.model.bert.embeddings.word_embeddings.weight.data[30500] = self.model.bert.embeddings.word_embeddings.weight.data[101]
        
        if args.fp16:
            self.model.half()
            
#         if args.model_name == 'gcn_only_with_bert_embedding':    # Freeze the BERT model and train only the GCN module.
#             for name, p in self.model.named_parameters():
#                 if name.startswith('bert'):
#                     p.requires_grad = False
            
        n_trainable_params_bert, n_trainable_params_gcn, n_nontrainable_params = 0, 0, 0
        for n, p in self.model.named_parameters():
            n_params = torch.prod(torch.tensor(p.shape)) 
            if p.requires_grad and n.startswith('bert'):
                n_trainable_params_bert += n_params
            elif p.requires_grad and n.startswith('bert') == False:
                n_trainable_params_gcn += n_params
            elif p.requires_grad == False:
                n_nontrainable_params += n_params
        print('n_BERT_trainable_params: {0}, n_GCN_trainable_params: {1}, n_nontrainable_params: {2}'.format(n_trainable_params_bert, n_trainable_params_gcn, n_nontrainable_params))
            
        self.model.to(args.device)

        if self.opt.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.opt.gpu_id)

        # Prepare optimizer
        if args.fp16:
            self.param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                                    for n, param in self.model.named_parameters()]
        elif args.optimize_on_cpu:
            self.param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                                    for n, param in self.model.named_parameters()]
        else:
            self.param_optimizer = list(self.model.named_parameters())
            
            
        no_decay = ['bias', 'gamma', 'beta']
        if args.model_name in ['bert_td', 'fc', 'bert_gcls', 'bert_gcls_m1']:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.param_optimizer if n not in no_decay],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in self.param_optimizer if n in no_decay],
                 'weight_decay_rate': 0.0}
            ]
            
            self.optimizer = BERTAdam(optimizer_grouped_parameters,
                                      lr=args.learning_rate,
                                      warmup=args.warmup_proportion,
                                      t_total=self.num_train_steps)
            
            self.optimizer_gcn = None
        
        elif args.model_name in ['td_bert_with_gcn', 'bert_fc_gcn']:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.param_optimizer if n not in no_decay and n.startswith('bert')],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in self.param_optimizer if n in no_decay and n.startswith('bert')],
                 'weight_decay_rate': 0.0}
            ]
            
            optimizer_grouped_parameters_names_bert = [
                {'params': [n for n, p in self.param_optimizer if n not in no_decay and n.startswith('bert')],
                 'weight_decay_rate': 0.01},
                {'params': [n for n, p in self.param_optimizer if n in no_decay and n.startswith('bert')],
                 'weight_decay_rate': 0.0}
            ]
            
            print('-'*77)
            print('names of parameters in the BERTAdam optimizer: ')
            print(optimizer_grouped_parameters_names_bert)
            
            self.optimizer = BERTAdam(optimizer_grouped_parameters,
                                      lr=args.learning_rate,
                                      warmup=args.warmup_proportion,
                                      t_total=self.num_train_steps)
            
            print('names of parameters in the Adam optimizer: ')
            optimizer_grouped_parameters_names_gcn= {'params': [pname for pname, p in self.param_optimizer if not pname.startswith('bert')]}
            print(optimizer_grouped_parameters_names_gcn)
            print('-'*77)
            
            self.optimizer_gcn = torch.optim.Adam(
                [{'params': [p for pname, p in self.param_optimizer if not pname.startswith('bert')]}], lr=0.001,
                weight_decay=0.00001)    
            
        self.global_step = 0
        self.max_test_acc_INC = 0
        self.max_test_acc_rand = 0
        self.max_validation_acc_INC = 0
        self.max_validation_f1_INC = 0
        
        self.max_test_f1_INC = 0
        self.max_test_f1_rand = 0
        self.latest_increase_epoch = 0
        
        self.best_L_config_acc = []
        
        self.best_L_config_f1 = []
        
    ###############################################################################################
    ###############################################################################################
    
    def get_random_L_config(self):
        x = random.sample([2,3,4,5,6,7,8,9,10], 3)
        x.sort()
        return [0 for item in range(x[0])]+[1 for item in range(x[0], x[1])]+[2 for item in range(x[1], x[2])] + [3 for item in range(x[2], 12)]
    
        
    ###############################################################################################
    ###############################################################################################
    
    def do_train(self):  
        # for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        print('# of train_examples: ', len(self.dataset.train_examples))
        print('# of eval_examples: ', len(self.dataset.eval_examples))
        if task_name == 'mams':
            print('# of validation_examples: ', len(self.dataset.validation_examples))
        for i_epoch in range(int(args.num_train_epochs)):
            print('>' * 100)
            print('>' * 100)
            print('epoch: ', i_epoch)
            tr_loss = 0
            train_accuracy = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            y_pred = []
            y_true = []
            for step, batch in enumerate(tqdm(self.dataset.train_dataloader, desc="Training")):
                # batch = tuple(t.to(self.opt.device) for t in batch)
                
                if step % 100 == 0:
                    print('='*77)
                    print('compare word embeddings')
                    print(torch.sum(self.model.bert.embeddings.word_embeddings.weight.data[101]))
                    print(torch.sum(self.model.bert.embeddings.word_embeddings.weight.data[30500]))
                    
                self.model.train()
                self.optimizer.zero_grad()
                if self.optimizer_gcn != None:
                    self.optimizer_gcn.zero_grad()
                
                input_ids, input_mask, segment_ids, label_ids, all_input_guids = batch
                               
#                 input_ids = input_ids.to(self.opt.device)
                segment_ids = segment_ids.to(self.opt.device)
                input_mask = input_mask.to(self.opt.device)
                label_ids = label_ids.to(self.opt.device)
#                 train_extended_attention_mask = list(self.train_extended_attention_mask[all_input_guids].transpose(0,1))
                
                if self.opt.model_name in ['bert_gcls', 'bert_gcls_m1', 'bert_td']:
                    input_ids = input_ids.to(self.opt.device)
                    train_extended_attention_mask = self.train_extended_attention_mask[all_input_guids].transpose(0,1)
                    train_VDC_info = self.train_VDC_info[all_input_guids]
                    
                    train_AAW_mask_token = self.train_AAW_mask_token[all_input_guids]
                    
                    train_current_VDC = self.train_current_VDC[all_input_guids]
                    
#                     train_VDC_weight = self.train_VDC_weight[all_input_guids]
                
                tran_indices = []
                span_indices = []
                gcls_attention_mask = []
                
                scls_attention_mask = []
                for item in all_input_guids:
                    tran_indices.append(self.train_tran_indices[item])
                    span_indices.append(self.train_span_indices[item])
                
                if self.global_step % 50 == 0:
                    print('-'*77)
                    print(tokenizer.convert_ids_to_tokens(input_ids[0][:100]))
                    print('-'*77)
                    print('segment_ids: ')
                    print(segment_ids[0][:100])
                    print('input_mask: ')
                    print(input_mask[0][:100])
                    print('guid: ', all_input_guids[0])
                    x = (train_VDC_info[0] == 1).nonzero(as_tuple=True)[0]
                    print('train_VDC_info[0] target: ', tokenizer.convert_ids_to_tokens(input_ids[0][x]))
#                     x = (train_VDC_weight[0] == 1).nonzero(as_tuple=True)[0]
#                     print('train_VDC_weight[0] target: ', tokenizer.convert_ids_to_tokens(input_ids[0][x]))
                    x = (train_VDC_info[0] == -1).nonzero(as_tuple=True)[0]
                    print('train_VDC_info[0] second g: ', tokenizer.convert_ids_to_tokens(input_ids[0][x]))
                    print('train_extended_attention_mask layer 1')
                    y = (train_VDC_info[0] == 1000).nonzero(as_tuple=True)[0]
                    x = (train_extended_attention_mask[0][0][0][y[0]] == 0).nonzero(as_tuple=True)[0]
                    print(tokenizer.convert_ids_to_tokens(input_ids[0][x]))
                    print('train_extended_attention_mask layer 5')
                    x = (train_extended_attention_mask[4][0][0][y[0]] == 0).nonzero(as_tuple=True)[0]
                    print(tokenizer.convert_ids_to_tokens(input_ids[0][x]))
                    print('train_extended_attention_mask layer 9')
                    x = (train_extended_attention_mask[8][0][0][y[0]] == 0).nonzero(as_tuple=True)[0]
                    print(tokenizer.convert_ids_to_tokens(input_ids[0][x]))
                    print('train_current_VDC[0]: ', train_current_VDC[0])
                
                
                if self.opt.model_class in [BertForSequenceClassification, CNN]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids)
                    
#                 elif self.opt.model_class in [BertForSequenceClassification_TD, CNN]:
#                     loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, VDC_info = train_VDC_info)
                    
                    #####
                elif self.opt.model_class in [BertForSequenceClassification_gcls]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, train_extended_attention_mask,
                                              g_pooler = args.g_pooler, VDC_info = train_VDC_info,
                                              AAW_mask_token = train_AAW_mask_token, current_VDC = train_current_VDC)
                    #####
                elif self.opt.model_class in [BertForSequenceClassification_gcls_m1]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, 
                                              extended_attention_mask = train_extended_attention_mask, 
                                              VDC_info = train_VDC_info, heuristic_vdc_auto = args.heuristic_vdc_auto)
                else:
                    input_t_ids = input_t_ids.to(self.opt.device)
                    input_t_mask = input_t_mask.to(self.opt.device)
                    segment_t_ids = segment_t_ids.to(self.opt.device)
                    if self.opt.model_class == MemNet:
                        input_without_t_ids = input_without_t_ids.to(self.opt.device)
                        input_without_t_mask = input_without_t_mask.to(self.opt.device)
                        segment_without_t_ids = segment_without_t_ids.to(self.opt.device)
                        loss, logits = self.model(input_without_t_ids, segment_without_t_ids, input_without_t_mask,
                                                  label_ids, input_t_ids, input_t_mask, segment_t_ids)
                    elif self.opt.model_class in [Cabasc]:
                        input_left_t_ids = input_left_t_ids.to(self.opt.device)
                        input_left_t_mask = input_left_t_mask.to(self.opt.device)
                        segment_left_t_ids = segment_left_t_ids.to(self.opt.device)
                        input_right_t_ids = input_right_t_ids.to(self.opt.device)
                        input_right_t_mask = input_right_t_mask.to(self.opt.device)
                        segment_right_t_ids = segment_right_t_ids.to(self.opt.device)
                        loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids,
                                                  input_t_ids, input_t_mask, segment_t_ids,
                                                  input_left_t_ids, input_left_t_mask, segment_left_t_ids,
                                                  input_right_t_ids, input_right_t_mask, segment_right_t_ids)
                    elif self.opt.model_class in [RAM, TNet_LF, MGAN, MLP, TD_BERT, TD_BERT_QA, DTD_BERT]:
                        input_left_ids = input_left_ids.to(self.opt.device)
                        input_left_mask = input_left_mask.to(self.opt.device)
                        segment_left_ids = segment_left_ids.to(self.opt.device)
                        loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids,
                                                  input_t_ids, input_t_mask, segment_t_ids,
                                                  input_left_ids, input_left_mask, segment_left_ids)
                    elif self.opt.model_class in [TD_BERT_with_GCN, BERT_FC_GCN]:
                        input_left_ids = input_left_ids.to(self.opt.device)
                        input_left_mask = input_left_mask.to(self.opt.device)
                        segment_left_ids = segment_left_ids.to(self.opt.device)
                        all_input_dg = all_input_dg.to(self.opt.device)
                        all_input_dg1 = all_input_dg1.to(self.opt.device)
                        loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids,
                                                  input_t_ids, input_t_mask, segment_t_ids,
                                                  input_left_ids, input_left_mask, segment_left_ids,
                                                  all_input_dg, all_input_dg1, tran_indices, span_indices)
                    else:
                        loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, input_t_ids,
                                                  input_t_mask, segment_t_ids)

                if self.opt.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                loss.backward()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                # 计算准确率
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_train_accuracy = accuracy(logits, label_ids)
                y_pred.extend(np.argmax(logits, axis=1))
                y_true.extend(label_ids)
                train_accuracy += tmp_train_accuracy
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in self.model.parameters():
                                param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(self.param_optimizer, self.model.named_parameters(),
                                                           test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            self.model.zero_grad()
                            continue
                        self.optimizer.step()
                        # self.optimizer_me.step()
                        copy_optimizer_params_to_model(self.model.named_parameters(), self.param_optimizer)
                    else:
                        self.optimizer.step()
                        if self.optimizer_gcn != None:
                            self.optimizer_gcn.step()
                        # self.optimizer_me.step()
                    self.model.zero_grad()
                    self.global_step += 1
                if self.global_step % self.opt.log_step == 0 and i_epoch > -1:
                    train_accuracy_ = train_accuracy / nb_tr_examples
                    train_f1 = f1_score(y_true, y_pred, average='macro', labels=np.unique(y_true))
                    
                    if task_name == 'mams':
                        result, validation_increased = self.do_eval('validation')
                        if validation_increased:
                            self.latest_increase_epoch = i_epoch
#                         if validation_increased:
#                             self.latest_increase_epoch = i_epoch
#                             print('='*77)
#                             print('Validation increased... testing on the test set')
#                             print('='*77)
#                             test_result, _ = self.do_eval()
#                             print('-'*77)
#                             print(f"Test results => Acc: {test_result['eval_accuracy']}, f1: {test_result['eval_f1']}")
#                             print('-'*77)
#                             self.last_test_acc = test_result['eval_accuracy']
#                             self.last_test_f1 = test_result['eval_f1']
#                             self.last_test_epoch = i_epoch
                    else:
                        result, validation_increased = self.do_eval()
                        if validation_increased:
                                self.latest_increase_epoch = i_epoch
                    tr_loss = tr_loss / nb_tr_steps
                    # self.scheduler.step(result['eval_accuracy'])
#                     self.writer.add_scalar('train_loss', tr_loss, i_epoch)
#                     self.writer.add_scalar('train_accuracy', train_accuracy_, i_epoch)
#                     self.writer.add_scalar('eval_accuracy', result['eval_accuracy'], i_epoch)
#                     self.writer.add_scalar('eval_loss', result['eval_loss'], i_epoch)
#                     self.writer.add_scalar('lr', self.optimizer_me.param_groups[0]['lr'], i_epoch)
                    
                    if task_name == 'mams':
#                         print(
#                         "Results: train_acc: {0:.6f} | train_f1: {1:.6f} | train_loss: {2:.6f} | eval_accuracy: {3:.6f} | eval_loss: {4:.6f} | eval_f1: {5:.6f} | max_validation_acc: {6:.6f} | max_validation_f1: {7:.6f} | max_test_acc: {8:.6f} | max_test_f1: {9:.6f} | last_test_acc: {10:.6f} | last_test_f1: {11:.6f} | last_test_epoch: {12:.6f} | latest_increase_epoch: {13:.1f}".format(
#                             train_accuracy_, train_f1, tr_loss, result['eval_accuracy'], result['eval_loss'], result['eval_f1'], self.max_validation_acc_INC, self.max_validation_f1_INC, self.max_test_acc_INC, self.max_test_f1_INC, self.last_test_acc, self.last_test_f1, self.last_test_epoch, self.latest_increase_epoch))
                        
                        print(
                        "Results: train_acc: {0:.6f} | train_f1: {1:.6f} | train_loss: {2:.6f} | eval_accuracy: {3:.6f} | eval_loss: {4:.6f} | eval_f1: {5:.6f} | max_validation_acc: {6:.6f} | max_validation_f1: {7:.6f} | latest_increase_epoch: {8:.1f}".format(
                            train_accuracy_, train_f1, tr_loss, result['eval_accuracy'], result['eval_loss'], result['eval_f1'], self.max_validation_acc_INC, self.max_validation_f1_INC, self.latest_increase_epoch))
                    else:
                        print(
                        "Results: train_acc: {0:.6f} | train_f1: {1:.6f} | train_loss: {2:.6f} | eval_accuracy: {3:.6f} | eval_loss: {4:.6f} | eval_f1: {5:.6f} | max_test_acc: {6:.6f} | max_test_f1: {7:.6f} | latest_increase_epoch: {8:.1f}".format(
                            train_accuracy_, train_f1, tr_loss, result['eval_accuracy'], result['eval_loss'], result['eval_f1'], self.max_test_acc_INC, self.max_test_f1_INC, self.latest_increase_epoch))
                        
    def do_eval(self, data_type = None):  
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        # confidence = []
        y_pred = []
        y_true = []
        
        if data_type == 'validation':
            d_loader = self.dataset.validation_dataloader
        else:
            d_loader = self.dataset.eval_dataloader
            
            
        for batch in tqdm(d_loader, desc="Evaluating"):
            # batch = tuple(t.to(self.opt.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, all_input_guids = batch
            
            segment_ids = segment_ids.to(self.opt.device)
            input_mask = input_mask.to(self.opt.device)
            label_ids = label_ids.to(self.opt.device)
            
            if self.opt.model_name in ['bert_gcls', 'bert_gcls_m1', 'bert_td']:
                input_ids = input_ids.to(self.opt.device)
                if data_type == 'validation':
                    extended_att_mask = self.validation_extended_attention_mask[all_input_guids].transpose(0,1)
                    eval_VDC_info = self.validation_VDC_info[all_input_guids]
                    
                    eval_AAW_mask_token = self.validation_AAW_mask_token[all_input_guids]
                    
                    eval_current_VDC = self.validation_current_VDC[all_input_guids]
                    
#                     eval_VDC_weight = self.validation_VDC_weight[all_input_guids]
                else:
                    extended_att_mask = self.eval_extended_attention_mask[all_input_guids].transpose(0,1)
                    eval_VDC_info = self.eval_VDC_info[all_input_guids]
                
                    eval_AAW_mask_token = self.eval_AAW_mask_token[all_input_guids]
                    
                    eval_current_VDC = self.eval_current_VDC[all_input_guids]
#                     eval_VDC_weight = self.eval_VDC_weight[all_input_guids]
            
#             if data_type == 'validation':
#                 extended_att_mask = list(self.validation_extended_attention_mask[all_input_guids].transpose(0,1))
#             else:
#                 extended_att_mask = list(self.eval_extended_attention_mask[all_input_guids].transpose(0,1))
            
            tran_indices = []
            span_indices = []
            gcls_attention_mask = []
            scls_attention_mask = []
            for item in all_input_guids:
                tran_indices.append(self.eval_tran_indices[item])
                span_indices.append(self.eval_span_indices[item])
                    
            with torch.no_grad():
                if self.opt.model_class in [BertForSequenceClassification, CNN]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids)
#                 elif self.opt.model_class in [BertForSequenceClassification_TD]:
#                     loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, VDC_info = eval_VDC_info)
                elif self.opt.model_class in [BertForSequenceClassification_gcls]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, extended_att_mask,
                                              g_pooler = args.g_pooler, VDC_info = eval_VDC_info, 
                                              AAW_mask_token = eval_AAW_mask_token, current_VDC = eval_current_VDC)
                elif self.opt.model_class in [BertForSequenceClassification_gcls_m1]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids,
                                              extended_attention_mask = extended_att_mask,
                                              VDC_info = eval_VDC_info, heuristic_vdc_auto = args.heuristic_vdc_auto)
                    
                else:
                    input_t_ids = input_t_ids.to(self.opt.device)
                    input_t_mask = input_t_mask.to(self.opt.device)
                    segment_t_ids = segment_t_ids.to(self.opt.device)
                    if self.opt.model_class == MemNet:
                        input_without_t_ids = input_without_t_ids.to(self.opt.device)
                        input_without_t_mask = input_without_t_mask.to(self.opt.device)
                        segment_without_t_ids = segment_without_t_ids.to(self.opt.device)
                        loss, logits = self.model(input_without_t_ids, segment_without_t_ids, input_without_t_mask,
                                                  label_ids, input_t_ids, input_t_mask, segment_t_ids)
                    elif self.opt.model_class in [Cabasc]:
                        input_left_t_ids = input_left_t_ids.to(self.opt.device)
                        input_left_t_ids = input_left_t_ids.to(self.opt.device)
                        input_left_t_mask = input_left_t_mask.to(self.opt.device)
                        segment_left_t_ids = segment_left_t_ids.to(self.opt.device)
                        input_right_t_ids = input_right_t_ids.to(self.opt.device)
                        input_right_t_mask = input_right_t_mask.to(self.opt.device)
                        segment_right_t_ids = segment_right_t_ids.to(self.opt.device)
                        loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids,
                                                  input_t_ids, input_t_mask, segment_t_ids,
                                                  input_left_t_ids, input_left_t_mask, segment_left_t_ids,
                                                  input_right_t_ids, input_right_t_mask, segment_right_t_ids)
                    elif self.opt.model_class in [RAM, TNet_LF, MGAN, MLP, TD_BERT, TD_BERT_QA, DTD_BERT]:
                        input_left_ids = input_left_ids.to(self.opt.device)
                        input_left_mask = input_left_mask.to(self.opt.device)
                        segment_left_ids = segment_left_ids.to(self.opt.device)
                        loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids,
                                                  input_t_ids, input_t_mask, segment_t_ids,
                                                  input_left_ids, input_left_mask, segment_left_ids)
                    elif self.opt.model_class in [TD_BERT_with_GCN, BERT_FC_GCN]:
                        input_left_ids = input_left_ids.to(self.opt.device)
                        input_left_mask = input_left_mask.to(self.opt.device)
                        segment_left_ids = segment_left_ids.to(self.opt.device)
                        all_input_dg = all_input_dg.to(self.opt.device)
                        all_input_dg1 = all_input_dg1.to(self.opt.device)
                        loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids,
                                                  input_t_ids, input_t_mask, segment_t_ids,
                                                  input_left_ids, input_left_mask, segment_left_ids,
                                                  all_input_dg, all_input_dg1, tran_indices, span_indices)
                    else:
                        loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, input_t_ids,
                                                  input_t_mask, segment_t_ids)

            # with torch.no_grad():  
            #     if self.opt.model_class in [BertForSequenceClassification, CNN]:
            #         loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids)
            #     else:
            #         loss, logits = self.model(input_ids, segment_ids, input_mask, labels=label_ids,
            #                                   input_t_ids=input_t_ids,
            #                                   input_t_mask=input_t_mask, segment_t_ids=segment_t_ids)
                    # confidence.extend(torch.nn.Softmax(dim=1)(logits)[:, 1].tolist())  # 获取 positive 类的置信度
            # loss = F.cross_entropy(logits, label_ids, size_average=False)  # 计算mini-batch的loss总和
            if self.opt.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.fp16 and args.loss_scale != 1.0:
                # rescale loss for fp16 training
                # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            logits = logits.detach().cpu().numpy()
            if self.opt.random_eval == True:
                for i in range(self.opt.random_eval_num):
                    logits_rand[i] = logits_rand[i].detach().cpu().numpy()
            
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
            if self.opt.random_eval == True:
                tmp_eval_accuracy_rand = [0] * self.opt.random_eval_num
                for i in range(self.opt.random_eval_num):
                    tmp_eval_accuracy_rand[i] = accuracy(logits_rand[i], label_ids)
            
            y_pred.extend(np.argmax(logits, axis=1))
            if self.opt.random_eval == True:
                for i in range(self.opt.random_eval_num):
                    y_pred_rand[i].extend(np.argmax(logits_rand[i], axis=1))
            
            y_true.extend(label_ids)

            # eval_loss += tmp_eval_loss.mean().item()
            eval_loss += loss.item()
            if self.opt.random_eval == True:
                for i in range(self.opt.random_eval_num):
                    eval_loss_rand[i] += loss_rand[i].item()
                    
            eval_accuracy += tmp_eval_accuracy
            if self.opt.random_eval == True:
                for i in range(self.opt.random_eval_num):
                    eval_accuracy_rand[i] += tmp_eval_accuracy_rand[i]

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        # eval_loss = eval_loss / len(self.dataset.eval_examples)
        test_f1 = f1_score(y_true, y_pred, average='macro', labels=np.unique(y_true))
        
        eval_loss = eval_loss / nb_eval_steps
        
        eval_accuracy = eval_accuracy / nb_eval_examples
               
        validation_increased = False
        if data_type == 'validation':
            if eval_accuracy > self.max_validation_acc_INC:
                validation_increased = True
                self.max_validation_acc_INC = eval_accuracy
                if self.max_test_acc_INC > 0.797 and self.opt.do_save == True:
                    torch.save(self.model.state_dict(), self.opt.model_save_path+'/best_acc.pkl')
                    print('='*77)
                    print('model saved at: ', self.opt.model_save_path + '/best_acc.pkl')
                    print('='*77)
            if test_f1 > self.max_validation_f1_INC:
                validation_increased = True
                self.max_validation_f1_INC = test_f1
                if self.max_test_f1_INC > 0.758 and self.opt.do_save == True:
                    torch.save(self.model.state_dict(), self.opt.model_save_path+'/best_f1.pkl')
                    print('='*77)
                    print('model saved at: ', self.opt.model_save_path + '/best_f1.pkl')
                    print('='*77)
        else:
            if eval_accuracy > self.max_test_acc_INC:
                validation_increased = True
                self.max_test_acc_INC = eval_accuracy
                if self.max_test_acc_INC > 0.797 and self.opt.do_save == True:
                    torch.save(self.model.state_dict(), self.opt.model_save_path+'/best_acc.pkl')
                    print('='*77)
                    print('model saved at: ', self.opt.model_save_path + '/best_acc.pkl')
                    print('='*77)
            if test_f1 > self.max_test_f1_INC:
                validation_increased = True
                self.max_test_f1_INC = test_f1
                if self.max_test_f1_INC > 0.758 and self.opt.do_save == True:
                    torch.save(self.model.state_dict(), self.opt.model_save_path+'/best_f1.pkl')
                    print('='*77)
                    print('model saved at: ', self.opt.model_save_path + '/best_f1.pkl')
                    print('='*77)
                
        
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'eval_f1': test_f1, }
         
        return result, validation_increased


    def do_predict(self):
        # 加载保存的模型进行预测，获得准确率
        # 读测试集的数据
        # dataset = ReadData(self.opt)  # 这个方法有点冗余了，读取了所有的数据，包括训练集
        # Load model
        saved_model = torch.load(self.opt.model_save_path)
        saved_model.to(self.opt.device)
        saved_model.eval()
        nb_test_examples = 0
        test_accuracy = 0
        for batch in tqdm(self.dataset.eval_dataloader, desc="Testing"):
            batch = tuple(t.to(self.opt.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, input_t_ids, input_t_mask, segment_t_ids = batch

            with torch.no_grad():  # Do not calculate gradient
                if self.opt.model_class in [BertForSequenceClassification, CNN]:
                    _, logits = saved_model(input_ids, segment_ids, input_mask, label_ids)
                else:
                    _, logits = saved_model(input_ids, segment_ids, input_mask, labels=label_ids,
                                            input_t_ids=input_t_ids,
                                            input_t_mask=input_t_mask, segment_t_ids=segment_t_ids)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_test_accuracy = accuracy(logits, label_ids)
            test_accuracy += tmp_test_accuracy
            nb_test_examples += input_ids.size(0)
        test_accuracy = test_accuracy / nb_test_examples
        return test_accuracy


    def run(self):
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

        self.do_train()
        print('>' * 100)
        if self.opt.do_predict:
            test_accuracy = self.do_predict()
            print("Test Set Accuracy: {:.4f}".format(test_accuracy))
        print("Max validate Set Acc_INC: {:.4f}, F1: {:.4f}".format(self.max_test_acc_INC, self.max_test_f1_INC))
        print("Max validate Set Acc_rand: {:.4f}, F1: {:.4f}".format(self.max_test_acc_rand, self.max_test_f1_rand))  
#         self.writer.close()
#         return self.max_test_acc
        return self.max_test_acc_INC, self.max_test_f1_INC


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    args = get_config()  # Gets the user settings or default hyperparameters
    processors = {
        "restaurant": RestaurantProcessor,
        "laptop": LaptopProcessor,
        "tweet": TweetProcessor,
        "mams": MamsProcessor,
    }
    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    args.processor = processors[task_name]()

    model_classes = {
        'cnn': CNN,
        'fc': BertForSequenceClassification,
        'clstm': CLSTM,
        'pf_cnn': PF_CNN,
        'tcn': TCN,
        'bert_pf': Bert_PF,
        'bbfc': BBFC,
        'tc_cnn': TC_CNN,
        'ram': RAM,
        'ian': IAN,
        'atae_lstm': ATAE_LSTM,
        'aoa': AOA,
        'memnet': MemNet,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'mgan': MGAN,
        'bert_ian': BERT_IAN,
        'tc_swem': TC_SWEM,
#         'tt': TT,
        'mlp': MLP,
        'aen': AEN_BERT,
        'td_bert': TD_BERT,
        'td_bert_with_gcn': TD_BERT_with_GCN,
        'bert_gcls': BertForSequenceClassification_gcls,
        'bert_td': BertForSequenceClassification_TD,
        'bert_gcls_m1': BertForSequenceClassification_gcls_m1,
        'bert_fc_gcn': BERT_FC_GCN,
        'td_bert_qa': TD_BERT_QA,
        'dtd_bert': DTD_BERT,
    }
    args.model_class = model_classes[args.model_name.lower()]

    if args.local_rank == -1 or args.no_cuda:  # if use multiple Gpus or no Gpus
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        while ',' in args.gpu_id:
            args.gpu_id.remove(',')
        args.gpu_id = list(map(int, args.gpu_id))
        args.n_gpu = len(args.gpu_id)
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            # torch.distributed.init_process_group(backend='nccl')
    logger.info(
        "device: {}, n_gpu: {}, distributed training: {}".format(args.device, args.n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:  # Gradient accumulation in distributed cases
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    ins = Instructor(args)
    max_test_acc, max_test_f1 = ins.run()

#     with open('result.txt', 'a', encoding='utf-8') as f:
#         f.write(str(max_test_acc)+ ',' + str(max_test_f1) + '\n')


    # result = []
    # for i in range(10):  # 跑10次，计算均值和标准差
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     if args.n_gpu > 0:
    #         torch.cuda.manual_seed_all(args.seed)
    #     print("Time: " + str(i))
    #     ins = Instructor(args)
    #     max_test_acc = ins.run()
    #     result.append(max_test_acc)
    # print(">"*100)
    # for i in result:
    #     print(i)
    # print(">"*100)
    # max_mean = np.mean(np.array(result))  # 计算均值
    # std = np.std(np.array(result).astype(float), ddof=1)  # 除以 n-1
    # std_2 = np.std(np.array(result).astype(float), ddof=0)  # 除以 n
    # print('均值：{:.4f}, 样本标准差： {:.4f}, 总体标准差: {:.4f}'.format(max_mean, std, std_2))
