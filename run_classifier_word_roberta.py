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
from utils.data_util_roberta import ReadData, RestaurantProcessor, LaptopProcessor, TweetProcessor
from utils.save_and_load import load_model_MoE, load_model_roberta_rpt
import torch.nn.functional as F
from sklearn.metrics import f1_score

import time
from data_utils import *
from transformers_ import BertTokenizer, RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, RobertaForSequenceClassification_gcls, RobertaForSequenceClassification_gcls_2, RobertaForSequenceClassification_lcf, RobertaModel, RobertaForSequenceClassification_TD, RobertaForSequenceClassification_gcls_td, RobertaForSequenceClassification_lcf_td, RobertaForSequenceClassification_asc_td, RobertaForSequenceClassification_gcls_att, RobertaForSequenceClassification_gcls_avg, RobertaForSequenceClassification_gcls_max

from torch.distributions.bernoulli import Bernoulli

from torch.optim.lr_scheduler import LambdaLR

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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
        roberta_config = RobertaConfig.from_json_file(args.roberta_config_file) 
        if args.max_seq_length > roberta_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                    args.max_seq_length, roberta_config.max_position_embeddings))

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
        
        if args.model_name in ['gcls', 'scls','roberta_gcls','roberta_td', 'roberta_lcf_td', 'roberta_gcls_td',
                               'roberta_asc_td']:
            self.train_extended_attention_mask = self.dataset.train_extended_attention_mask.to(args.device)
            self.eval_extended_attention_mask = self.dataset.eval_extended_attention_mask.to(args.device)
            
            
        if args.model_name in ['roberta_lcf', 'roberta_lcf_td']:
            self.train_lcf_vec_list = self.dataset.train_lcf_vec_list
            self.eval_lcf_vec_list = self.dataset.eval_lcf_vec_list
            
        print("label size: {}".format(args.output_dim))

        # 初始化模型
        print("initialize model ...")
        
        os.makedirs(args.model_save_path, exist_ok=True)
        
        if args.model_class == BertForSequenceClassification:
            self.model = BertForSequenceClassification(bert_config, len(self.dataset.label_list))
        elif args.model_class == RobertaForSequenceClassification:
#             self.model = RobertaForSequenceClassification(roberta_config)
            self.model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    
        elif args.model_class == RobertaForSequenceClassification_gcls:
            if args.g_pooler == 'att':
                self.model = RobertaForSequenceClassification_gcls_att.from_pretrained('roberta-base')
            elif args.g_pooler == 'avg':
                self.model = RobertaForSequenceClassification_gcls_avg.from_pretrained('roberta-base')
            elif args.g_pooler == 'max':
                self.model = RobertaForSequenceClassification_gcls_max.from_pretrained('roberta-base')
            
        elif args.model_class == RobertaForSequenceClassification_gcls_2:
            self.model = RobertaForSequenceClassification_gcls_2.from_pretrained('roberta-base')
        elif args.model_class == RobertaForSequenceClassification_gcls_td:
            self.model = RobertaForSequenceClassification_gcls_td.from_pretrained('roberta-base')
            self.model.roberta_td = RobertaModel.from_pretrained("roberta-base")
        elif args.model_class == RobertaForSequenceClassification_asc_td:
            self.model = RobertaForSequenceClassification_asc_td.from_pretrained('roberta-base')
            self.model.roberta_td = RobertaModel.from_pretrained("roberta-base")
        elif args.model_class == RobertaForSequenceClassification_lcf:
            self.model = RobertaForSequenceClassification_lcf.from_pretrained('roberta-base')
            self.model.roberta_local = RobertaModel.from_pretrained("roberta-base")
        elif args.model_class == RobertaForSequenceClassification_lcf_td:
            self.model = RobertaForSequenceClassification_lcf_td.from_pretrained('roberta-base')
            self.model.roberta_local = RobertaModel.from_pretrained("roberta-base")
        elif args.model_class == RobertaForSequenceClassification_TD:
            self.model = RobertaForSequenceClassification_TD.from_pretrained('roberta-base')
        elif args.model_class == BertForSequenceClassification_GCLS:
            self.model = BertForSequenceClassification_GCLS(bert_config, len(self.dataset.label_list))
        else:
            self.model = model_classes[args.model_name](bert_config, args)
        
        if self.opt.model_name in ['gcls_moe']:
            self.model = load_model_MoE(self.model, self.opt.init_checkpoint, self.opt.init_checkpoint_2, 
                                        self.opt.init_checkpoint_3, self.opt.init_checkpoint_4, self.opt.init_checkpoint_5)
            print('-'*77)
            print('1st MoE BERT loading from ', self.opt.init_checkpoint)
            print('2nd MoE BERT loading from ', self.opt.init_checkpoint_2)
            print('3rd MoE BERT loading from ', self.opt.init_checkpoint_3)
            print('4th MoE BERT loading from ', self.opt.init_checkpoint_4)
            print('5th MoE BERT loading from ', self.opt.init_checkpoint_5)
            
            print('-'*77)
        elif self.opt.model_name in ['roberta_gcls_moe']:
            self.model = load_model_roberta_rpt(self.model, self.opt.init_checkpoint, self.opt.init_checkpoint_2, 
                                        self.opt.init_checkpoint_3, self.opt.init_checkpoint_4)
        else:
            if args.model_name not in ['roberta', 'roberta_gcls', 'roberta_gcls_2', 'roberta_gcls_td', 'roberta_lcf', 'robeta_lcf_td', 'roberta_td', 'roberta_asc_td'] and 'pytorch_model.bin' in self.opt.init_checkpoint:
                self.model.load_state_dict(torch.load(self.opt.init_checkpoint, map_location='cpu'))
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
        
        # do it only once for rpt!
#         torch.save(self.model.state_dict(), './roberta_rpt/init_ckpt.pkl')
        
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
        if args.model_name in ['td_bert', 'fc', 'gcls', 'scls', 'gcls_moe']:
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
        
        elif args.model_name in ['roberta', 'roberta_gcls', 'roberta_gcls_2', 'roberta_gcls_td', 'roberta_lcf', 'roberta_lcf_td','roberta_td', 'roberta_asc_td']:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.param_optimizer if n not in no_decay],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in self.param_optimizer if n in no_decay],
                 'weight_decay_rate': 0.0}
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
            scheduler = LambdaLR(self.optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
            
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
        
        self.max_test_f1_INC = 0
        self.max_test_f1_rand = 0
        
        self.best_L_config_acc = []
        
        self.best_L_config_f1 = []
        

        
    ###############################################################################################
    ###############################################################################################
    
    def get_random_L_config(self):
        x = random.sample([2,3,4,5,6,7,8,9,10], 2)
        x.sort()
        return [0 for item in range(x[0])]+[1 for item in range(x[0], x[1])]+[2 for item in range(x[1], 12)] 
        
    ###############################################################################################
    ###############################################################################################
    
    def do_train(self):
        
        self.train_g_config = []
        if len(self.opt.g_config) == 4:
            for i in range(12):
                self.train_g_config.append(torch.tensor([[self.opt.g_config[0], self.opt.g_config[1]],
                                                         [self.opt.g_config[2], self.opt.g_config[3]]], dtype = torch.float))
        
        elif len(self.opt.g_config) == 9:
            for i in range(12):
                if i < self.opt.g_config[8]:
                    self.train_g_config.append(torch.tensor([[self.opt.g_config[0], self.opt.g_config[1]],
                                                         [self.opt.g_config[2], self.opt.g_config[3]]], dtype = torch.float))
                else:
                    self.train_g_config.append(torch.tensor([[self.opt.g_config[4], self.opt.g_config[5]],
                                                         [self.opt.g_config[6], self.opt.g_config[7]]], dtype = torch.float))
                    
        elif len(self.opt.g_config) == 12:
            for i in range(12):
                if i < 4:
                    self.train_g_config.append(torch.tensor([[self.opt.g_config[0], self.opt.g_config[1]],
                                                         [self.opt.g_config[2], self.opt.g_config[3]]], dtype = torch.float))
                elif i < 8:
                    self.train_g_config.append(torch.tensor([[self.opt.g_config[4], self.opt.g_config[5]],
                                                         [self.opt.g_config[6], self.opt.g_config[7]]], dtype = torch.float)) 
                elif i < 12:
                    self.train_g_config.append(torch.tensor([[self.opt.g_config[8], self.opt.g_config[9]],
                                                         [self.opt.g_config[10], self.opt.g_config[11]]], dtype = torch.float))
                
        print('# of train_examples: ', len(self.dataset.train_examples))
        print('# of eval_examples: ', len(self.dataset.eval_examples))
        
        train_layer_L = self.opt.L_config_base
        train_layer_L_set = set(train_layer_L)
        
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
                
                input_ids, label_ids, all_input_guids = batch
                if self.opt.model_name in ['roberta_lcf', 'roberta_lcf_td']:
                    input_ids_lcf_global = input_ids_lcf_global.to(self.opt.device)
                    input_ids_lcf_local = input_ids_lcf_local.to(self.opt.device)
                elif self.opt.model_name in ['roberta_gcls_td', 'roberta_asc_td']:
                    input_ids_lcf_global = input_ids_lcf_global.to(self.opt.device)
                    input_ids_lcf_local = input_ids_lcf_local.to(self.opt.device)
                elif self.opt.model_name in ['roberta_gcls']:
                    input_ids = input_ids.to(self.opt.device)
                    train_extended_attention_mask = list(self.train_extended_attention_mask[all_input_guids].transpose(0,1))
                else:
                    input_ids = input_ids.to(self.opt.device)
                    segment_ids = segment_ids.to(self.opt.device)
                    input_mask = input_mask.to(self.opt.device)
                    
                label_ids = label_ids.to(self.opt.device)
                                        
                if self.opt.model_name in ['roberta_lcf', 'roberta_lcf_td']:
                    train_lcf_matrix = torch.tensor([self.train_lcf_vec_list[item] for item in all_input_guids], 
                                                    dtype = torch.float)
                    train_lcf_matrix = train_lcf_matrix.to(self.opt.device)
                
                if self.global_step % 5000000 == 499999:
                    if self.opt.model_name in ['gcls', 'gcls_moe', 'roberta_gcls',
                                              'roberta_td']:
                        print('-'*77)
                        print(tokenizer.convert_ids_to_tokens(input_ids[0][:50]))
                        print('guid: ', all_input_guids[0])
                        print('graph_s_pos[0]: ', graph_s_pos[0])
                        print('graph_s tokens are: ', tokenizer.convert_ids_to_tokens(input_ids[0][graph_s_pos[0]==1]))
                        print('gcls_attention_mask[0][:5]: ', train_gcls_attention_mask[0][:5])
                        
                        
                    elif self.opt.model_name in ['roberta_gcls_2']:
                        print('-'*77)
                        print(tokenizer.convert_ids_to_tokens(input_ids[0][:50]))
                        print('segment_ids: ', segment_ids[0][:50])
                        print('input_mask: ', input_mask[0][:50])
                        print('guid: ', all_input_guids[0])
                        print('gcls_attention_mask.size(): ' ,gcls_attention_mask.size())
                        print('gcls_attention_mask[0][0]: ', gcls_attention_mask[0][0])
                        print('gcls_attention_mask_2[0][0]: ', gcls_attention_mask_2[0][0])
                        print('-'*77)
                        print('gcls_attention_mask[0][1]: ', gcls_attention_mask[0][1])
                        print('gcls_attention_mask_2[0][1]: ', gcls_attention_mask_2[0][1])
                        print('-'*77)
                        print('gcls_attention_mask[0][2]: ', gcls_attention_mask[0][2])
                        print('gcls_attention_mask_2[0][2]: ', gcls_attention_mask_2[0][2])
                        print('-'*77)
                        
                    elif self.opt.model_name in ['roberta_lcf', 'roberta_lcf_td']:
                        print('-'*77)
                        print('input_ids_lcf_global[0][:50]: ')
                        print(tokenizer.convert_ids_to_tokens(input_ids_lcf_global[0][:50]))
                        print('input_ids_lcf_local[0][:50]: ')
                        print(tokenizer.convert_ids_to_tokens(input_ids_lcf_local[0][:50]))
                        print('train_lcf_matrix[0]: ', train_lcf_matrix[0])
                        print('gcls_attention_mask[0][:50]: ', gcls_attention_mask[0][:50])
                       
                    elif self.opt.model_name in ['roberta_asc_td']:
                        print('-'*77)
                        print('input_ids_lcf_global[0][:50]: ')
                        print(tokenizer.convert_ids_to_tokens(input_ids_lcf_global[0][:50]))
                        print('input_ids_lcf_local[0][:50]: ')
                        print(tokenizer.convert_ids_to_tokens(input_ids_lcf_local[0][:50]))
                      
                    elif self.opt.model_name in ['roberta']:
                        print('-'*77)
                        print('input_ids[0][:50]: ')
                        print(tokenizer.convert_ids_to_tokens(input_ids[0][:50]))
                        
                    elif self.opt.model_name in ['roberta_gcls_td']:
                        print('-'*77)
                        print('input_ids_lcf_global[0][:50]: ')
                        print(tokenizer.convert_ids_to_tokens(input_ids_lcf_global[0][:50]))
                        print('input_ids_lcf_local[0][:50]: ')
                        print(tokenizer.convert_ids_to_tokens(input_ids_lcf_local[0][:50]))
                        print('gcls_attention_mask[0][:50]: ', gcls_attention_mask[0][:50])
                
                
                if self.opt.model_class in [BertForSequenceClassification, CNN]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids)
                    
                elif self.opt.model_class in [RobertaForSequenceClassification]:
                    loss, logits = self.model(input_ids, labels = label_ids)[:2]
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_gcls]:
                    loss, logits = self.model(input_ids, labels = label_ids,
                                              extended_attention_mask = train_extended_attention_mask)[:2]
                
                elif self.opt.model_class in [RobertaForSequenceClassification_gcls_2]:
                    loss, logits = self.model(input_ids, labels = label_ids, gcls_attention_mask = gcls_attention_mask, 
                                              gcls_attention_mask_2 = gcls_attention_mask_2,
                                              layer_L=train_layer_L, layer_L_2 = train_layer_L_2, g_config = self.train_g_config, 
                                              g_token_pos = self.opt.g_token_pos)[:2]
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_gcls_td]:
                    loss, logits = self.model(input_ids_lcf_global, input_ids_lcf_local, labels = label_ids, 
                                              gcls_attention_mask = gcls_attention_mask, layer_L=train_layer_L, 
                                              g_config = self.train_g_config, g_token_pos = self.opt.g_token_pos, 
                                              target_idx = torch.tensor(train_target_idx))[:2]
                
                elif self.opt.model_class in [RobertaForSequenceClassification_asc_td]:
                    loss, logits = self.model(input_ids_lcf_global, input_ids_lcf_local, labels = label_ids, 
                                              gcls_attention_mask = gcls_attention_mask)[:2]
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_lcf]:
                    loss, logits = self.model(input_ids_lcf_global, input_ids_lcf_local, labels = label_ids, 
                                              lcf_matrix = train_lcf_matrix)[:2]
                
                elif self.opt.model_class in [RobertaForSequenceClassification_lcf_td]:
                    loss, logits = self.model(input_ids_lcf_global, input_ids_lcf_local, labels = label_ids, 
                                              lcf_matrix = train_lcf_matrix, gcls_attention_mask = gcls_attention_mask)[:2]
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_TD]:
                    loss, logits = self.model(input_ids, labels = label_ids, gcls_attention_mask = train_gcls_attention_mask)[:2]
                    
                elif self.opt.model_class in [BertForSequenceClassification_GCLS]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, gcls_attention_mask,
                                              train_layer_L)
                elif self.opt.model_class in [BertForSequenceClassification_GCLS_MoE]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids, gcls_attention_mask,
                                              train_layer_L, MoE_layer)
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
                    print('lr: ', self.optimizer.param_groups[0]['lr'])
                    train_accuracy_ = train_accuracy / nb_tr_examples
                    train_f1 = f1_score(y_true, y_pred, average='macro', labels=np.unique(y_true))
                    result = self.do_eval()
                    tr_loss = tr_loss / nb_tr_steps
                    # self.scheduler.step(result['eval_accuracy'])
#                     self.writer.add_scalar('train_loss', tr_loss, i_epoch)
#                     self.writer.add_scalar('train_accuracy', train_accuracy_, i_epoch)
#                     self.writer.add_scalar('eval_accuracy', result['eval_accuracy'], i_epoch)
#                     self.writer.add_scalar('eval_loss', result['eval_loss'], i_epoch)
#                     self.writer.add_scalar('lr', self.optimizer_me.param_groups[0]['lr'], i_epoch)
                    
                    if self.opt.random_eval == False:
                        print(
                        "Results: train_acc: {0:.6f} | train_f1: {1:.6f} | train_loss: {2:.6f} | eval_accuracy: {3:.6f} | eval_loss: {4:.6f} | eval_f1: {5:.6f} | max_test_acc: {6:.6f} | max_test_f1: {7:.6f}".format(
                            train_accuracy_, train_f1, tr_loss, result['eval_accuracy'], result['eval_loss'], result['eval_f1'], self.max_test_acc_INC, self.max_test_f1_INC))
                    else:
                        print(
                            "Results: train_acc: {0:.6f} | train_f1: {1:.6f} | train_loss: {2:.6f} | eval_accuracy: {3:.6f} | eval_loss: {4:.6f} | eval_f1: {5:.6f}".format(
                                train_accuracy_, train_f1, tr_loss, result['eval_accuracy'], result['eval_loss'], result['eval_f1']))
                        print()
                        print(" | max_test_acc_INC: {0:.6f} | max_test_f1_INC: {1:.6f}".format(self.max_test_acc_INC,
                                                                                               self.max_test_f1_INC))
                        print(" | max_test_acc_rand: {0:.6f} | max_test_f1_rand: {1:.6f}".format(self.max_test_acc_rand,
                                                                                                 self.max_test_f1_rand))
                        print(f" | best_L_config_acc: {self.best_L_config_acc} |")
                        print(f" | best_L_config_f1: {self.best_L_config_f1} |")
                        
    def do_eval(self):  
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        # confidence = []
        y_pred = []
        y_true = []
        
        layer_L = self.opt.L_config_base
        layer_L_set = set(layer_L)
        
        for batch in tqdm(self.dataset.eval_dataloader, desc="Evaluating"):
            # batch = tuple(t.to(self.opt.device) for t in batch)
            input_ids, label_ids, all_input_guids = batch
                
                
            if self.opt.model_name in ['roberta_lcf', 'roberta_lcf_td']:
                input_ids_lcf_global = input_ids_lcf_global.to(self.opt.device)
                input_ids_lcf_local = input_ids_lcf_local.to(self.opt.device)
                
            elif self.opt.model_name in ['roberta_gcls_td', 'roberta_asc_td']:
                input_ids_lcf_global = input_ids_lcf_global.to(self.opt.device)
                input_ids_lcf_local = input_ids_lcf_local.to(self.opt.device)
            elif self.opt.model_name in ['roberta_gcls']:
                input_ids = input_ids.to(self.opt.device)
                eval_extended_attention_mask = list(self.eval_extended_attention_mask[all_input_guids].transpose(0,1))
            else:
                input_ids = input_ids.to(self.opt.device)
                segment_ids = segment_ids.to(self.opt.device)
                input_mask = input_mask.to(self.opt.device)

            label_ids = label_ids.to(self.opt.device)
            
            tran_indices = []
            span_indices = []
                    
            with torch.no_grad():
                if self.opt.model_class in [BertForSequenceClassification, CNN]:
                    loss, logits = self.model(input_ids, segment_ids, input_mask, label_ids)
                    
                elif self.opt.model_class in [RobertaForSequenceClassification]:
                    loss, logits = self.model(input_ids, labels = label_ids)[:2]
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_gcls]:
                    loss, logits = self.model(input_ids, labels = label_ids,
                                              extended_attention_mask = eval_extended_attention_mask)[:2]
                  
                elif self.opt.model_class in [RobertaForSequenceClassification_gcls_2]:
                    loss, logits = self.model(input_ids, labels = label_ids, gcls_attention_mask=gcls_attention_mask,
                                              gcls_attention_mask_2=gcls_attention_mask_2,
                                              layer_L=layer_L, layer_L_2 = layer_L_2, g_config = self.train_g_config, 
                                              g_token_pos = self.opt.g_token_pos,
                                              target_idx = torch.tensor(eval_target_idx))[:2]
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_gcls_td]:
                    loss, logits = self.model(input_ids_lcf_global, input_ids_lcf_local, labels = label_ids, 
                                              gcls_attention_mask=gcls_attention_mask, layer_L=layer_L, 
                                              g_config = self.train_g_config, g_token_pos = self.opt.g_token_pos,
                                              target_idx = torch.tensor(eval_target_idx))[:2]
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_asc_td]:
                    loss, logits = self.model(input_ids_lcf_global, input_ids_lcf_local, labels = label_ids, 
                                              gcls_attention_mask = gcls_attention_mask )[:2]
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_lcf]:
                    loss, logits = self.model(input_ids_lcf_global, input_ids_lcf_local, labels = label_ids, 
                                              lcf_matrix = eval_lcf_matrix)[:2]
                
                elif self.opt.model_class in [RobertaForSequenceClassification_lcf_td]:
                    loss, logits = self.model(input_ids_lcf_global, input_ids_lcf_local, labels = label_ids, 
                                              lcf_matrix = eval_lcf_matrix, gcls_attention_mask = gcls_attention_mask)[:2]
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_TD]:
                    loss, logits = self.model(input_ids, labels = label_ids, gcls_attention_mask = eval_gcls_attention_mask)[:2]
                            
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
            
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
            
            y_pred.extend(np.argmax(logits, axis=1))
            
            y_true.extend(label_ids)

            # eval_loss += tmp_eval_loss.mean().item()
            eval_loss += loss.item()
                    
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        # eval_loss = eval_loss / len(self.dataset.eval_examples)
        test_f1 = f1_score(y_true, y_pred, average='macro', labels=np.unique(y_true))
        
        eval_loss = eval_loss / nb_eval_steps
        
        eval_accuracy = eval_accuracy / nb_eval_examples
        
        if eval_accuracy > self.max_test_acc_INC:
            self.max_test_acc_INC = eval_accuracy
            if self.max_test_acc_INC > 0.797 and self.opt.do_save == True:
                torch.save(self.model.state_dict(), self.opt.model_save_path+'/best_acc.pkl')
                print('='*77)
                print('model saved at: ', self.opt.model_save_path + '/best_acc.pkl')
                print('='*77)
        if test_f1 > self.max_test_f1_INC:
            self.max_test_f1_INC = test_f1
            if self.max_test_f1_INC > 0.758 and self.opt.do_save == True:
                torch.save(self.model.state_dict(), self.opt.model_save_path+'/best_f1.pkl')
                print('='*77)
                print('model saved at: ', self.opt.model_save_path + '/best_f1.pkl')
                print('='*77)
                
        if self.opt.random_eval == False:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'eval_f1': test_f1, }
        else:
            result = {'eval_loss': eval_loss,
                      'eval_accuracy': eval_accuracy,
                      'eval_f1': test_f1, }
            for i in range(self.opt.random_eval_num):
                result['eval_loss_'+str(i+1)] = eval_loss_rand[i]
                result['eval_accuracy_'+str(i+1)] = eval_accuracy_rand[i]
                result['eval_f1_'+str(i+1)] = eval_accuracy_rand[i]
            
        return result


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
        'roberta': RobertaForSequenceClassification,
        'roberta_gcls': RobertaForSequenceClassification_gcls,
        'roberta_gcls_2': RobertaForSequenceClassification_gcls_2,
        'roberta_gcls_td': RobertaForSequenceClassification_gcls_td,
        'roberta_asc_td': RobertaForSequenceClassification_asc_td,
        'roberta_lcf': RobertaForSequenceClassification_lcf,
        'roberta_lcf_td': RobertaForSequenceClassification_lcf_td,
        'roberta_td': RobertaForSequenceClassification_TD,
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
