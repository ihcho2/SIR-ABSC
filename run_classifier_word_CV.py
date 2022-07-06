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
from modeling import BertConfig, BertForSequenceClassification, BertForSequenceClassification_GCLS, BertForSequenceClassification_GCLS_MoE
from optimization import BERTAdam

from configs import get_config
from models import CNN, CLSTM, PF_CNN, TCN, Bert_PF, BBFC, TC_CNN, RAM, IAN, ATAE_LSTM, AOA, MemNet, Cabasc, TNet_LF, MGAN, BERT_IAN, TC_SWEM, MLP, AEN_BERT, TD_BERT, TD_BERT_QA, DTD_BERT, TD_BERT_with_GCN, BERT_FC_GCN
from utils.data_util import ReadData, RestaurantProcessor, LaptopProcessor, TweetProcessor
from utils.save_and_load import load_model_MoE
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
        
        if args.model_name in ['gcls', 'scls', 'gcls_er', 'gcls_moe']:
            self.train_gcls_attention_mask = self.dataset.train_gcls_attention_mask
            
            self.eval_gcls_attention_mask = self.dataset.eval_gcls_attention_mask
            
        if args.model_name in ['scls']:
            self.train_scls_input_mask = self.dataset.train_scls_input_mask
            self.eval_scls_input_mask = self.dataset.eval_scls_input_mask

        
        print("label size: {}".format(args.output_dim))

        # 初始化模型
        print("initialize model ...")
        
        os.makedirs(args.model_save_path, exist_ok=True)
        
        if args.model_class == BertForSequenceClassification:
            self.model = BertForSequenceClassification(bert_config, len(self.dataset.label_list))
        elif args.model_class == BertForSequenceClassification_GCLS:
            self.model = BertForSequenceClassification_GCLS(bert_config, len(self.dataset.label_list))
        elif args.model_class == BertForSequenceClassification_GCLS_MoE:
            self.model = BertForSequenceClassification_GCLS_MoE(bert_config, len(self.dataset.label_list))
        else:
            self.model = model_classes[args.model_name](bert_config, args)
        
        if self.opt.model_name in ['gcls_moe']:
            self.model = load_model_MoE(self.model, self.opt.init_checkpoint, self.opt.init_checkpoint_2, 
                                        self.opt.init_checkpoint_3, self.opt.init_checkpoint_4)
            print('-'*77)
            print('1st MoE BERT loading from ', self.opt.init_checkpoint)
            print('2nd MoE BERT loading from ', self.opt.init_checkpoint_2)
            print('3rd MoE BERT loading from ', self.opt.init_checkpoint_3)
            print('4th MoE BERT loading from ', self.opt.init_checkpoint_4)
            
            print('-'*77)
        else:
            if 'pytorch_model.bin' in self.opt.init_checkpoint:
                self.model.bert.load_state_dict(torch.load(self.opt.init_checkpoint, map_location='cpu'))
                print('-'*77)
                print('Loading from ', self.opt.init_checkpoint)
                print('-'*77)
        
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
        if args.model_name in ['td_bert', 'fc', 'gcls', 'scls', 'gcls_er', 'gcls_moe']:
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
        self.max_train_eval_acc_INC = 0
        self.max_train_eval_acc_rand = 0
        self.max_test_acc = 0
        
        self.max_train_eval_f1_INC = 0
        self.max_train_eval_f1_rand = 0
        self.max_test_f1 = 0
        
        self.best_L_config_acc = []
        self.best_L_config_f1 = []
        
        ############### Checking several assertions for each training mode.
        
        if self.opt.model_name in ['fc']:
            self.opt.random_config_training = False
            self.opt.random_eval = False
            
        elif self.opt.model_name in ['gcls']:
            if self.opt.random_config_training == False:
                assert len(self.opt.L_config_base) == 12
            
        elif self.opt.model_name in ['gcls_moe', 'gcls_moe_default']:
            self.opt.random_config_training = True
        
        ###############
        
    ###############################################################################################
    ###############################################################################################
    
    def get_random_L_config(self):
        x = random.sample([2,3,4,5,6,7,8,9,10], 3)
        x.sort()
        return [0 for item in range(x[0])]+[1 for item in range(x[0], x[1])]+[2 for item in range(x[1], x[2])]+[3 for item in range(x[2], 12)]
    
        
    ###############################################################################################
    ###############################################################################################
    
    def do_train(self):  # 训练模型
        # for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        print('# of train_examples: ', len(self.dataset.train_examples))
        print('# of eval_examples: ', len(self.dataset.eval_examples))
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
                self.model.train()
                self.optimizer.zero_grad()
                if self.optimizer_gcn != None:
                    self.optimizer_gcn.zero_grad()
                
                input_ids, input_mask, segment_ids, label_ids, \
                all_input_dg, all_input_dg1, all_input_dg2, all_input_dg3, all_input_guids = batch
                               
                input_ids = input_ids.to(self.opt.device)
                segment_ids = segment_ids.to(self.opt.device)
                input_mask = input_mask.to(self.opt.device)
                label_ids = label_ids.to(self.opt.device)
                
                tran_indices = []
                span_indices = []
                gcls_attention_mask = []
                
                scls_attention_mask = []
                for item in all_input_guids:
                    tran_indices.append(self.train_tran_indices[item])
                    span_indices.append(self.train_span_indices[item])
                    if self.opt.model_name in ['gcls', 'scls', 'gcls_er', 'gcls_moe']:
                        gcls_attention_mask.append(self.train_gcls_attention_mask[2][item])
                    if self.opt.model_name in ['scls']:
                        scls_attention_mask.append(self.train_scls_input_mask[item])
                
                if self.global_step % 50 == 0:
                    print('-'*77)
                    print(tokenizer.convert_ids_to_tokens(input_ids[0][:50]))
                    print('segment_ids: ', segment_ids[0][:50])
                    print('input_mask: ', input_mask[0][:50])
                    print('guid: ', all_input_guids[0])
                    if self.opt.model_name in ['gcls', 'gcls_moe']:
                        print('gcls_attention_mask[0][:50]: ', gcls_attention_mask[0][:50])
                
                
                ###############################################################################################
                ###############################################################################################

#                 if self.opt.random_config_training == True:
#                     if i_epoch > self.opt.rct_warmup - 1:
#                         layer_L = self.get_random_L_config()

#                     else:
#                         layer_L = self.opt.L_config_base
#                 else:
#                     layer_L = self.opt.L_config_base

                if self.opt.random_config_training == True:
                    if self.global_step %2 == 100:
#                         layer_L = [0,0,0,0,1,1,1,1,2,2,2,2]
                        layer_L = self.opt.L_config_base
                        MoE_layer = random.choices([0,1,2,3], k = 12)
                    else:
                        layer_L = self.get_random_L_config()
#                         layer_L = [0,0,0,0,1,1,1,1,2,2,2,2]
                        MoE_layer = random.choices([0,1,2,3], k = 12)
                else:
                    layer_L = self.opt.L_config_base
                    
                ###############################################################################################
                ###############################################################################################
                
                
                if self.opt.model_class in [BertForSequenceClassification, CNN]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids)
                elif self.opt.model_class in [BertForSequenceClassification_GCLS]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, gcls_attention_mask,
                                              layer_L)
                elif self.opt.model_class in [BertForSequenceClassification_GCLS_MoE]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, gcls_attention_mask,
                                              layer_L, MoE_layer)
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
                if self.global_step % self.opt.log_step == 0:
                    train_accuracy_ = train_accuracy / nb_tr_examples
                    train_f1 = f1_score(y_true, y_pred, average='macro', labels=np.unique(y_true))
                    should_test_L_config_, should_test_MoE_layer_ = self.do_eval(self.dataset.train_eval_dataloader)
                    
                    for ii in range(len(should_test_L_config_)):
                        _,_ = self.do_eval(self.dataset.eval_dataloader, L_config_ = should_test_L_config_[ii],
                                           MoE_layer_ = should_test_MoE_layer_[ii])  
                    
                    tr_loss = tr_loss / nb_tr_steps
                    
                    if self.opt.random_eval == False:
                        print(
                        "Results: train_acc: {0:.6f} | train_f1: {1:.6f} | train_loss: {2:.6f} | eval_accuracy: {3:.6f} | eval_loss: {4:.6f} | eval_f1: {5:.6f} | max_test_acc: {6:.6f} | max_test_f1: {7:.6f}".format(
                            train_accuracy_, train_f1, tr_loss, result['eval_accuracy'], result['eval_loss'], result['eval_f1'], self.max_test_acc_INC, self.max_test_f1_INC))
                    else:
#                         print(
#                             "Results: train_acc: {0:.6f} | train_f1: {1:.6f} | train_loss: {2:.6f} | eval_accuracy: {3:.6f} | eval_loss: {4:.6f} | eval_f1: {5:.6f}".format(
#                                 train_accuracy_, train_f1, tr_loss, result['eval_accuracy'], result['eval_loss'], result['eval_f1']))
                        print()
                        print(" | max_train_eval_acc_rand: {0:.6f} | max_train_eval_f1_rand:{1:.6f}".format(self.max_train_eval_acc_rand, self.max_train_eval_f1_rand))
                        print(" | max_test_acc: {0:.6f} | max_test_f1: {1:.6f}".format(self.max_test_acc,
                                                                                               self.max_test_f1))
                        print(f" | best_L_config_acc: {self.best_L_config_acc} |")
                        print(f" | best_MoE_layer_acc: {self.best_MoE_layer_acc} |")
                        
                        print(f" | best_L_config_f1: {self.best_L_config_f1} |")
                        print(f" | best_MoE_layer_f1: {self.best_MoE_layer_f1} |")
                        
                        
                self.best_MoE_layer_acc = MoE_layer
                        
    def do_eval(self, eval_dataloader_, L_config_= None, MoE_layer_ = None):  
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        # confidence = []
        y_pred = []
        y_true = []
        
        should_test_L_config = []
        should_test_MoE_layer = []
        
        if L_config_ == None:
            ###############################################################################################
            ############################################################################################### 
            
            layer_L = self.opt.L_config_base
            MoE_layer = random.choices([0,1,2,3], k=12)
            
            layer_L_rand = []
            MoE_layer_rand = []
            
            for i in range(self.opt.random_eval_num):
                layer_L_rand.append(self.get_random_L_config())
                MoE_layer_rand.append(random.choices([0,1,2,3], k=12))
                
            
            ###############################################################################################
            ###############################################################################################
            
            eval_loss_rand = [0] * self.opt.random_eval_num
            eval_accuracy_rand = [0] * self.opt.random_eval_num
            nb_eval_steps_rand = [0] * self.opt.random_eval_num
            nb_eval_examples_rand = [0] * self.opt.random_eval_num
            
            y_pred_rand = [[],[],[],[],[],[],[],[],[],[]]
            
        else:
            layer_L = L_config_
            MoE_layer = MoE_layer_
            print('-'*77)
            print('testing on the test set')
            print('eval_layer_L: ', layer_L)
            print('MoE_layer: ', MoE_layer)
            
        for batch in tqdm(eval_dataloader_, desc="Evaluating"):
            # batch = tuple(t.to(self.opt.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, \
            all_input_dg, all_input_dg1, all_input_dg2, all_input_dg3, all_input_guids = batch

            input_ids = input_ids.to(self.opt.device)
            segment_ids = segment_ids.to(self.opt.device)
            input_mask = input_mask.to(self.opt.device)
            label_ids = label_ids.to(self.opt.device)
            
            tran_indices = []
            span_indices = []
            gcls_attention_mask = []
            scls_attention_mask = []
            for item in all_input_guids:
                if self.opt.model_name in ['gcls', 'gcls_er', 'gcls_moe']:
                    if L_config_ == None:
                        gcls_attention_mask.append(self.train_gcls_attention_mask[2][item])
                    else:
                        gcls_attention_mask.append(self.eval_gcls_attention_mask[2][item])
                    
            with torch.no_grad():
                if self.opt.model_class in [BertForSequenceClassification, CNN]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids)
                elif self.opt.model_class in [BertForSequenceClassification_GCLS]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, gcls_attention_mask,
                                              layer_L)
                elif self.opt.model_class in [BertForSequenceClassification_GCLS_MoE]:
                    if L_config_ != None:
                        loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, gcls_attention_mask,
                                              layer_L, MoE_layer)
                    else:
                        loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, gcls_attention_mask,
                                              layer_L, MoE_layer)
                        
                        loss_rand = [0] * self.opt.random_eval_num
                        logits_rand = [0] * self.opt.random_eval_num
                        
                        for i in range(self.opt.random_eval_num):
                            loss_rand[i], logits_rand[i] = self.model(input_ids, segment_ids, input_mask, label_ids, 
                                                                      gcls_attention_mask, layer_L_rand[i], MoE_layer_rand[i])
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
            if L_config_ == None:
                for i in range(self.opt.random_eval_num):
                    logits_rand[i] = logits_rand[i].detach().cpu().numpy()
            
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
            if L_config_ == None:
                tmp_eval_accuracy_rand = [0] * self.opt.random_eval_num
                for i in range(self.opt.random_eval_num):
                    tmp_eval_accuracy_rand[i] = accuracy(logits_rand[i], label_ids)
            
            y_pred.extend(np.argmax(logits, axis=1))
            if L_config_ == None:
                for i in range(self.opt.random_eval_num):
                    y_pred_rand[i].extend(np.argmax(logits_rand[i], axis=1))
            
            y_true.extend(label_ids)

            # eval_loss += tmp_eval_loss.mean().item()
            eval_loss += loss.item()
            if L_config_ == None:
                for i in range(self.opt.random_eval_num):
                    eval_loss_rand[i] += loss_rand[i].item()
                    
            eval_accuracy += tmp_eval_accuracy
            if L_config_ == None:
                for i in range(self.opt.random_eval_num):
                    eval_accuracy_rand[i] += tmp_eval_accuracy_rand[i]

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        # eval_loss = eval_loss / len(self.dataset.eval_examples)
        test_f1 = f1_score(y_true, y_pred, average='macro', labels=np.unique(y_true))
        if L_config_ == None:
            test_f1_rand = [0] * self.opt.random_eval_num
            for i in range(self.opt.random_eval_num):
                test_f1_rand[i] = f1_score(y_true, y_pred_rand[i], average='macro', labels=np.unique(y_true))
        
        eval_loss = eval_loss / nb_eval_steps
        if L_config_ == None:
            for i in range(self.opt.random_eval_num):
                eval_loss_rand[i] = eval_loss_rand[i] / nb_eval_steps
        
        eval_accuracy = eval_accuracy / nb_eval_examples
        if L_config_ == None:
            for i in range(self.opt.random_eval_num):
                eval_accuracy_rand[i] = eval_accuracy_rand[i] / nb_eval_examples
        
        if L_config_ == None:    # means train_eval_dataloader
            if eval_accuracy > self.max_train_eval_acc_INC:
                self.max_train_eval_acc_INC = eval_accuracy
                should_test_L_config.append(self.opt.L_config_base)
                should_test_MoE_layer.append([0]*12)
                
            if test_f1 > self.max_train_eval_f1_INC:
                self.max_train_eval_f1_INC = test_f1
                should_test_L_config.append(self.opt.L_config_base)
                should_test_MoE_layer.append([0]*12)

            if self.opt.random_eval == True:
                for i in range(self.opt.random_eval_num):
                    if eval_accuracy_rand[i] > self.max_train_eval_acc_rand:
                        self.max_train_eval_acc_rand = eval_accuracy_rand[i]
                        should_test_L_config.append(layer_L_rand[i])
                        should_test_MoE_layer.append(MoE_layer_rand[i])
                        
                    if test_f1_rand[i] > self.max_train_eval_f1_rand:
                        self.max_train_eval_f1_rand = test_f1_rand[i]
                        should_test_L_config.append(layer_L_rand[i])
                        should_test_MoE_layer.append(MoE_layer_rand[i])
                        
        else:
            if eval_accuracy > self.max_test_acc:
                self.max_test_acc = eval_accuracy
                self.best_L_config_acc = L_config_
                self.best_MoE_layer_acc = MoE_layer_
                
            if test_f1 > self.max_test_f1:
                self.max_test_f1 = test_f1
                self.best_L_config_f1 = L_config_
                self.best_MoE_layer_f1 = MoE_layer_
                
        return should_test_L_config, should_test_MoE_layer


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
        'gcls': BertForSequenceClassification_GCLS,
        'gcls_moe': BertForSequenceClassification_GCLS_MoE,
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
