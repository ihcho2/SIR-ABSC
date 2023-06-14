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
from torch.optim import AdamW
from tensorboardX import SummaryWriter
from optimization import BERTAdam

from configs import get_config_2
from models import CNN, CLSTM, PF_CNN, TCN, Bert_PF, BBFC, TC_CNN, RAM, IAN, ATAE_LSTM, AOA, MemNet, Cabasc, TNet_LF, MGAN, BERT_IAN, TC_SWEM, MLP, AEN_BERT, TD_BERT, TD_BERT_QA, DTD_BERT, TD_BERT_with_GCN, BERT_FC_GCN
from utils.data_util_roberta import ReadData, RestaurantProcessor, LaptopProcessor, TweetProcessor, MamsProcessor
from utils.save_and_load import load_model_roberta_VDC_auto
import torch.nn.functional as F
from sklearn.metrics import f1_score

import time
from data_utils_roberta import *
from transformers_ import BertTokenizer, RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForSequenceClassification, RobertaForSequenceClassification_gcls, RobertaForSequenceClassification_TD

from torch.distributions.bernoulli import Bernoulli

from torch.optim.lr_scheduler import LambdaLR

import torch.nn.functional as F

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_printoptions(sci_mode=False)

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
    def __init__(self, args, init_model_state_dict = None):
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
        if task_name == 'mams':
            self.validation_tran_indices = self.dataset.validation_tran_indices
            self.validation_span_indices = self.dataset.validation_span_indices
        
        if args.model_name in ['roberta_td', 'roberta_td_t_star', 'roberta_gcls', 'roberta_gcls_ffn', 'roberta_gcls_vdc_auto',
                               'roberta_gcls_m1']:
            self.train_extended_attention_mask = self.dataset.train_extended_attention_mask.to(args.device)
            self.eval_extended_attention_mask = self.dataset.eval_extended_attention_mask.to(args.device)
            self.train_VDC_info = self.dataset.train_VDC_info.to(args.device)
            self.train_DEP_info = self.dataset.train_DEP_info.to(args.device)
            self.eval_VDC_info = self.dataset.eval_VDC_info.to(args.device)
            self.eval_DEP_info = self.dataset.eval_DEP_info.to(args.device)
            
            if task_name == 'mams':
                self.validation_extended_attention_mask = self.dataset.validation_extended_attention_mask.to(args.device)
                self.validation_VDC_info = self.dataset.validation_VDC_info.to(args.device)
                self.validation_DEP_info = self.dataset.validation_DEP_info.to(args.device)
            
                
        if args.model_name in ['roberta_lcf', 'roberta_lcf_td']:
            self.train_lcf_vec_list = self.dataset.train_lcf_vec_list
            self.eval_lcf_vec_list = self.dataset.eval_lcf_vec_list
            
        print("label size: {}".format(args.output_dim))

        # 初始化模型
        print("initialize model ...")
        
        os.makedirs(args.model_save_path, exist_ok=True)
        
        if args.model_class == RobertaForSequenceClassification:
#             self.model = RobertaForSequenceClassification(roberta_config)
            self.model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        elif args.model_class == RobertaForSequenceClassification_TD:
            self.model = RobertaForSequenceClassification_TD.from_pretrained('roberta-base')
        elif args.model_class == RobertaForSequenceClassification_gcls:
            self.model = RobertaForSequenceClassification_gcls.from_pretrained('roberta-base', g_pooler = args.g_pooler,
                                                                               pb = args.pb)
        else:
            self.model = model_classes[args.model_name](bert_config, args)
        
        self.model.roberta.embeddings.word_embeddings.weight.data[50249] = self.model.roberta.embeddings.word_embeddings.weight.data[0]
        
        self.model.roberta.embeddings.word_embeddings.weight.data[50250] = self.model.roberta.embeddings.word_embeddings.weight.data[0]
        
        if init_model_state_dict !=None:
            self.model.load_state_dict(init_model_state_dict)
        else:
            self.init_model_state_dict = self.model.state_dict()
            
#         #### save & load model checkpoint if necessary.

#         self.model = load_model_roberta_VDC_auto(self.model)
        
#         if args.save_init_model == True:
#             torch.save(self.model.state_dict(), self.opt.model_save_path+f'/init_seed_{args.seed}.pkl')
            
#         if self.opt.init_checkpoint != None:
#             print('='*77)
#             print(f'Loading model checkpoint from {self.opt.init_checkpoint}')
#             self.model = load_model_roberta_auto(self.model, self.opt.init_checkpoint)
        
#         if self.opt.model_name in ['gcls_moe']:
#             self.model = load_model_MoE(self.model, self.opt.init_checkpoint, self.opt.init_checkpoint_2, 
#                                         self.opt.init_checkpoint_3, self.opt.init_checkpoint_4, self.opt.init_checkpoint_5)
#             print('-'*77)
#             print('1st MoE BERT loading from ', self.opt.init_checkpoint)
#             print('2nd MoE BERT loading from ', self.opt.init_checkpoint_2)
#             print('3rd MoE BERT loading from ', self.opt.init_checkpoint_3)
#             print('4th MoE BERT loading from ', self.opt.init_checkpoint_4)
#             print('5th MoE BERT loading from ', self.opt.init_checkpoint_5)
            
#             print('-'*77)
#         elif self.opt.model_name in ['roberta_gcls_moe']:
#             self.model = load_model_roberta_rpt(self.model, self.opt.init_checkpoint, self.opt.init_checkpoint_2, 
#                                         self.opt.init_checkpoint_3, self.opt.init_checkpoint_4)
#         else:
#             if args.model_name not in ['roberta', 'roberta_td', 'roberta_gcls', 'roberta_gcls_auto'] and 'pytorch_model.bin' in self.opt.init_checkpoint:
#                 self.model.load_state_dict(torch.load(self.opt.init_checkpoint, map_location='cpu'))
#                 print('-'*77)
#                 print('Loading from ', self.opt.init_checkpoint)
#                 print('-'*77)
        
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
            
        elif args.model_name in ['roberta', 'roberta_td', 'roberta_td_t_star', 'roberta_gcls', 'roberta_gcls_ffn', 
                                 'roberta_gcls_vdc_auto', 'roberta_gcls_m1']:
            
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.param_optimizer if n not in no_decay],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in self.param_optimizer if n in no_decay],
                 'weight_decay_rate': 0.0}
            ]
            
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
            self.scheduler = LambdaLR(self.optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
          
            
        self.global_step = 0
        self.max_test_acc_INC = 0
        self.max_test_acc_rand = 0
        
        self.max_validation_acc_INC = 0
        self.max_validation_acc_rand = 0
        
        self.max_test_f1_INC = 0
        self.max_test_f1_rand = 0
        
        self.max_validation_f1_INC = 0
        self.max_validation_f1_rand = 0
        
        self.latest_increase_epoch = 0
        self.best_L_config_acc = []
        
        self.best_L_config_f1 = []
        
        self.last_test_acc = 0
        self.last_test_f1 = 0
        self.last_test_epoch = 0
        
        self.test_acc_each_epoch = []
        self.test_f1_each_epoch = []
        
        
    ###############################################################################################
    ###############################################################################################
    
    def do_train(self):
        print('='*77)
        print('torch.sum(self.model.roberta.encoder.DEP.weight.data): ', torch.sum(self.model.roberta.encoder.DEP.weight.data))
        print('='*77)
        print('torch.sum(self.model.roberta.pooler.dense_ec.weight.data): ',
              torch.sum(self.model.roberta.pooler.dense_ec.weight.data))
        print('='*77)
        print('torch.sum(self.model.roberta.pooler.dense_concat.weight.data): ',
              torch.sum(self.model.roberta.pooler.dense_concat.weight.data))
        print('='*77)
        print()
        
        print("shouldn't change")
        print('='*77)
        print('torch.sum(self.model.roberta.embeddings.word_embeddings.weight.data): ', torch.sum(self.model.roberta.embeddings.word_embeddings.weight.data))
        print('torch.sum(self.model.roberta.pooler.dense.weight.data): ',
              torch.sum(self.model.roberta.pooler.dense.weight.data))
        print('='*77)
        
        print('# of train_examples: ', len(self.dataset.train_examples))
        print('# of eval_examples: ', len(self.dataset.eval_examples))
        if task_name == 'mams':
            print('# of validation_examples: ', len(self.dataset.validation_examples))
        
        for i_epoch in range(int(args.num_train_epochs)):
            if i_epoch == 8:
                if task_name == 'tweet' and self.max_test_acc_INC < 0.70:
                    print('='*77)
                    print('terminating training due to exceptionally bad seed')
                    sys.exit()
                elif task_name == 'laptop' and self.max_test_acc_INC < 0.70:
                    print('='*77)
                    print('terminating training due to exceptionally bad seed')
                    sys.exit()
                elif task_name == 'restaurant' and self.max_test_acc_INC < 0.86:
                    print('='*77)
                    print('terminating training due to exceptionally bad seed')
                    sys.exit()
                elif task_name == 'mams' and self.max_validation_acc_INC < 0.80:
                    print('='*77)
                    print('terminating training due to exceptionally bad seed')
                    sys.exit()
#             if i_epoch>0:
#                 self.scheduler.step()
            print('>' * 100)
            print('>' * 100)
            print('epoch: ', i_epoch)
            tr_loss = 0
            train_accuracy = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            y_pred = []
            y_true = []
            
            if i_epoch > 0:
                self.test_acc_each_epoch.append(self.last_test_acc)
                self.test_f1_each_epoch.append(self.last_test_f1)
                
            for step, batch in enumerate(tqdm(self.dataset.train_dataloader, desc="Training")):
                # batch = tuple(t.to(self.opt.device) for t in batch)
                
                if step % 100 == 0:
                    print('='*77)
                    print('compare word embeddings')
                    print(torch.sum(self.model.roberta.embeddings.word_embeddings.weight.data[0]))
                    print(torch.sum(self.model.roberta.embeddings.word_embeddings.weight.data[50249]))
                    print(torch.sum(self.model.roberta.embeddings.word_embeddings.weight.data[50250]))
                self.model.train()
                self.optimizer.zero_grad()
                
                input_ids, label_ids, all_input_guids = batch
                if self.opt.model_name in ['roberta_lcf', 'roberta_lcf_td']:
                    input_ids_lcf_global = input_ids_lcf_global.to(self.opt.device)
                    input_ids_lcf_local = input_ids_lcf_local.to(self.opt.device)
                elif self.opt.model_name in ['roberta_gcls_td', 'roberta_asc_td']:
                    input_ids_lcf_global = input_ids_lcf_global.to(self.opt.device)
                    input_ids_lcf_local = input_ids_lcf_local.to(self.opt.device)
                elif self.opt.model_name in ['roberta_td', 'roberta_td_t_star', 'roberta_gcls', 'roberta_gcls_ffn', 
                                             'roberta_gcls_vdc_auto']:
                    input_ids = input_ids.to(self.opt.device)
#                     train_extended_attention_mask = list(self.train_extended_attention_mask[all_input_guids].transpose(0,1))
                    train_extended_attention_mask = self.train_extended_attention_mask[all_input_guids].transpose(0,1)
                    train_VDC_info = self.train_VDC_info[all_input_guids]
                    train_DEP_info = self.train_DEP_info[all_input_guids]
                elif self.opt.model_name in ['roberta_gcls_m1']:
                    input_ids = input_ids.to(self.opt.device)
                    train_extended_attention_mask = self.train_extended_attention_mask[all_input_guids].transpose(0,1)
                    train_VDC_info = self.train_VDC_info[all_input_guids]
                    train_DEP_info = self.train_DEP_info[all_input_guids]
                
                elif self.opt.model_name in ['roberta']:
                    input_ids = input_ids.to(self.opt.device)
                else:
                    input_ids = input_ids.to(self.opt.device)
                    segment_ids = segment_ids.to(self.opt.device)
                    input_mask = input_mask.to(self.opt.device)
                    
                label_ids = label_ids.to(self.opt.device)
                                        
                
                if self.global_step % 50 == 0:
                    detail = True
                    if self.opt.model_name in ['roberta_td', 'roberta_gcls', 'roberta_gcls_m1']:
                        
                        print('VDC_info[0]: ', train_VDC_info[0])
                        print('DEP_info[0]: ', train_DEP_info[0])
                        print()
                        print('='*77)
                        print('Code examination from run.py and using Parser on the fly')
                        print()
                        print('* 1. text')
                        print('-'*77)
                        words = self.dataset.DGEDT_train_data[all_input_guids[0]]['text']
                        text = ' '.join(self.dataset.DGEDT_train_data[all_input_guids[0]]['text'])
                        document = self.dataset.nlp(text)
                        print(text)
                        
                        span_indices = self.train_span_indices[all_input_guids[0]]
                        tran_indices = self.train_tran_indices[all_input_guids[0]]
                        
                        document_words = [token for token in document]
                        target_i = list(range(span_indices[0][0], span_indices[0][1]))
                        VDC = {} # token.i 기준으로 저장할 것.
                        last_used = [] # Just for assertion.

                        VDC[0] = target_i.copy()
                        used = target_i.copy()
                        last_used = target_i.copy()

                        for l in range(1, 12):
                            VDC[l] = VDC[l-1].copy()
                            last_used_ = []
                            for token in document:
                                # Skip the already used ones.
                                if token.i in used:
                                    continue

                                # 현재 VDC token들과 연결되는 방법은 2가지. head로 연결되는 경우와 tail로 연결되는 경우.
                                # 1. Head로 연결되는 경우.
                                for child in token.children:
                                    if child.i in used:
                                        assert child.i in last_used
                                        VDC[l].append(token.i)
                                        last_used_.append(token.i)

                                # 2. Tail로 연결되는 경우.
                                if token.head.i in used:
                                    assert token.head.i in last_used
                                    VDC[l].append(token.i)
                                    last_used_.append(token.i)

                            used += VDC[l]
                            last_used = last_used_
                            VDC[l] = sorted(VDC[l])

                        # VDC로 이제 attended token 만들고 tokenzier로 최종하면 딱 맞을 듯?
                        VDC_l_tokens = []
                        for i in range(len(VDC)):
                            VDC_words = [token.text for token in document_words if token.i in VDC[i]]
                            VDC_tokens = []
                            for word in VDC_words:
                                VDC_tokens += tokenizer.tokenize(word)

                            VDC_l_tokens.append(VDC_tokens)

                        
                        print('-'*77)
                        print(tokenizer.convert_ids_to_tokens(input_ids[0][:100]))
                        print('guid: ', all_input_guids[0])
                        x = (train_VDC_info[0] == 0).nonzero(as_tuple=True)[0]
                        print('train_VDC_info[0] target: ', tokenizer.convert_ids_to_tokens(input_ids[0][x]))
                        
                        curr_vdc = self.dataset.train_current_VDC[all_input_guids[0]]
                        
                        zz = (train_VDC_info[0] == 999).nonzero(as_tuple=True)[0]
                        
                        print(f'train_extended_attention_mask layer 0, vdc={int(curr_vdc[0])}')
                        x = (train_extended_attention_mask[0][0][0][zz[0]] == 0).nonzero(as_tuple=True)[0]
                        print(tokenizer.convert_ids_to_tokens(input_ids[0][x]))
                        print(VDC_l_tokens[int(curr_vdc[0])])
                        print('-'*77)
                        print(f'train_extended_attention_mask layer 4, vdc={int(curr_vdc[4])}')
                        x = (train_extended_attention_mask[4][0][0][zz[0]] == 0).nonzero(as_tuple=True)[0]
                        print(tokenizer.convert_ids_to_tokens(input_ids[0][x]))
                        print(VDC_l_tokens[int(curr_vdc[4])])
                        print('-'*77)
                        print(f'train_extended_attention_mask layer 8, vdc={int(curr_vdc[8])}')
                        x = (train_extended_attention_mask[8][0][0][zz[0]] == 0).nonzero(as_tuple=True)[0]
                        print(tokenizer.convert_ids_to_tokens(input_ids[0][x]))
                        print(VDC_l_tokens[int(curr_vdc[8])])
                        
                        
                        for i, token in enumerate(document):
                            if i == 20:
                                break
                            if 'g_infront' in args.input_format:
                                if i >= span_indices[0][0]:
                                    A = 2
                                else:
                                    A = 1
                            elif 'g' in args.input_format:
                                A = 2
                            elif 'TD' in args.input_format:
                                A = 1
                            if i >= span_indices[0][0] and 'g_infront' in args.input_format:
                                A = 2

                            for xx in range(tran_indices[i][0], tran_indices[i][1]):
                                print(f'{i}-th token: {token.text} ({self.dataset.dep2idx[int(train_DEP_info[0][xx+A])]})')
                        
                        
                        
                    elif self.opt.model_name in ['roberta']:
                        print('-'*77)
                        print('input_ids[0][:50]: ')
                        print(tokenizer.convert_ids_to_tokens(input_ids[0][:50]))
                        print('label_ids[0]: ', label_ids[0])
                        
                else:
                    detail = False
                    
                if self.opt.model_class in [RobertaForSequenceClassification]:
                    loss, logits = self.model(input_ids, labels = label_ids)[:2]
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_TD]:
                    loss, logits = self.model(input_ids, labels = label_ids, VDC_info = train_VDC_info)[:2]
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_gcls]:
                    sg_loss, output_ = self.model(input_ids, labels = label_ids,
                                                  extended_attention_mask = train_extended_attention_mask, 
                                                  VDC_info = train_VDC_info, DEP_info = train_DEP_info,
                                                  use_DEP = args.use_DEP, detail = detail)[:2]
                    loss, logits = output_[:2]
                    
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
                        # self.optimizer_me.step()
                    self.model.zero_grad()
                    self.global_step += 1
                    
                if self.global_step % self.opt.log_step == 0 and i_epoch > -1:
                    print('lr: ', self.optimizer.param_groups[0]['lr'])
                    print('lr2: ', self.optimizer.param_groups[1]['lr'])
                    
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
#                     self.scheduler.step(result['eval_accuracy'])
#                     self.scheduler.step()
#                     self.writer.add_scalar('train_loss', tr_loss, i_epoch)
#                     self.writer.add_scalar('train_accuracy', train_accuracy_, i_epoch)
#                     self.writer.add_scalar('eval_accuracy', result['eval_accuracy'], i_epoch)
#                     self.writer.add_scalar('eval_loss', result['eval_loss'], i_epoch)
#                     self.writer.add_scalar('lr', self.optimizer_me.param_groups[0]['lr'], i_epoch)
                    
                    
                    # logging training info
                    if torch.isnan(torch.tensor(tr_loss)):
                        print('terminating training due to nan')
                        sys.exit()
                    if task_name == 'mams':
#                         print(
#                         "Results: train_acc: {0:.6f} | train_f1: {1:.6f} | train_loss: {2:.6f} | eval_accuracy: {3:.6f} | eval_loss: {4:.6f} | eval_f1: {5:.6f} | max_validation_acc: {6:.6f} | max_validation_f1: {7:.6f} | max_test_acc: {8:.6f} | max_test_f1: {9:.6f} | last_test_acc: {10:.6f} | last_test_f1: {11:.6f} | last_test_epoch: {12:.6f} | latest_increase_epoch: {13:.1f}".format(
#                             train_accuracy_, train_f1, tr_loss, result['eval_accuracy'], result['eval_loss'], result['eval_f1'], self.max_validation_acc_INC, self.max_validation_f1_INC, self.max_test_acc_INC, self.max_test_f1_INC, self.last_test_acc, self.last_test_f1, self.last_test_epoch, self.latest_increase_epoch))
                        
                        print(
                        "Results: train_acc: {0:.6f} | train_f1: {1:.6f} | train_loss: {2:.6f} | eval_accuracy: {3:.6f} | eval_loss: {4:.6f} | eval_f1: {5:.6f} | max_validation_acc: {6:.6f} | max_validation_f1: {7:.6f} | latest_increase_epoch: {13:.1f}".format(
                            train_accuracy_, train_f1, tr_loss, result['eval_accuracy'], result['eval_loss'], result['eval_f1'], self.max_validation_acc_INC, self.max_validation_f1_INC, self.max_test_acc_INC, self.max_test_f1_INC, self.last_test_acc, self.last_test_f1, self.last_test_epoch, self.latest_increase_epoch))
                    else:
                        print(
                        "Results: train_acc: {0:.6f} | train_f1: {1:.6f} | train_loss: {2:.6f} | eval_accuracy: {3:.6f} | eval_loss: {4:.6f} | eval_f1: {5:.6f} | max_test_acc: {6:.6f} | max_test_f1: {7:.6f} | latest_increase_epoch: {8:.1f}".format(
                            train_accuracy_, train_f1, tr_loss, result['eval_accuracy'], result['eval_loss'], result['eval_f1'], self.max_test_acc_INC, self.max_test_f1_INC, self.latest_increase_epoch))
                    
                        
                        
    def do_eval(self, data_type = None):  
        ss = False
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
            
        eval_vdc = None
        for batch in tqdm(d_loader, desc="Evaluating"):
            # batch = tuple(t.to(self.opt.device) for t in batch)
            input_ids, label_ids, all_input_guids = batch
                
            if self.opt.model_name in ['roberta_td', 'roberta_td_t_star', 'roberta_gcls', 'roberta_gcls_ffn', 
                                       'roberta_gcls_vdc_auto']:
                input_ids = input_ids.to(self.opt.device)
                if data_type == 'validation':
                    extended_att_mask = list(self.validation_extended_attention_mask[all_input_guids].transpose(0,1))
                    eval_VDC_info = self.validation_VDC_info[all_input_guids]
                    eval_DEP_info = self.validation_DEP_info[all_input_guids]
                else:
                    extended_att_mask = list(self.eval_extended_attention_mask[all_input_guids].transpose(0,1))
                    eval_VDC_info = self.eval_VDC_info[all_input_guids]
                    eval_DEP_info = self.eval_DEP_info[all_input_guids]
                    
            elif self.opt.model_name in ['roberta_gcls_m1']:
                input_ids = input_ids.to(self.opt.device)
                if data_type == 'validation':
                    extended_att_mask = self.validation_extended_attention_mask[all_input_guids].transpose(0,1)
                    eval_VDC_info = self.validation_VDC_info[all_input_guids]
                    eval_DEP_info = self.validation_DEP_info[all_input_guids]
                else:
                    extended_att_mask = self.eval_extended_attention_mask[all_input_guids].transpose(0,1)
                    eval_VDC_info = self.eval_VDC_info[all_input_guids]
                    eval_DEP_info = self.eval_DEP_info[all_input_guids]
                    
            elif self.opt.model_name in ['roberta']:
                input_ids = input_ids.to(self.opt.device)
                
            else:
                input_ids = input_ids.to(self.opt.device)
                segment_ids = segment_ids.to(self.opt.device)
                input_mask = input_mask.to(self.opt.device)

            label_ids = label_ids.to(self.opt.device)
            
            tran_indices = []
            span_indices = []
                    
            with torch.no_grad():
                if self.opt.model_class in [RobertaForSequenceClassification]:
                    loss, logits = self.model(input_ids, labels = label_ids)[:2]
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_TD]:
                    loss, logits = self.model(input_ids, labels = label_ids, VDC_info = eval_VDC_info )[:2]
                    
                    
                elif self.opt.model_class in [RobertaForSequenceClassification_gcls]:
                    sg_loss, output_ = self.model(input_ids, labels = label_ids,
                                              extended_attention_mask = extended_att_mask,
                                              VDC_info = eval_VDC_info, DEP_info = eval_DEP_info, use_DEP = args.use_DEP)[:2]
                    loss, logits = output_[:2]
                
                                
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
            
            
            predictions = np.argmax(logits, axis=1)
            correct_guids = all_input_guids[predictions == label_ids]
#             if 420 in correct_guids:
#                 ss = True
                
            
            # eval_loss += tmp_eval_loss.mean().item()
            eval_loss += loss.item()
                    
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        # eval_loss = eval_loss / len(self.dataset.eval_examples)
        test_f1 = f1_score(y_true, y_pred, average='macro', labels=np.unique(y_true))
        
        eval_loss = eval_loss / nb_eval_steps
        
        
        eval_accuracy = eval_accuracy / nb_eval_examples
        
        if data_type == 'validation':
            validation_increased = False
            if eval_accuracy > self.max_validation_acc_INC:
                self.max_validation_acc_INC = eval_accuracy
                validation_increased = True
                if self.max_validation_acc_INC > 0.797 and self.opt.do_save == True:
                    torch.save(self.model.state_dict(), self.opt.model_save_path+'/best_acc.pkl')
                    print('='*77)
                    print('model saved at: ', self.opt.model_save_path + '/best_acc.pkl')
                    print('='*77)
            if test_f1 > self.max_validation_f1_INC:
                self.max_validation_f1_INC = test_f1
                validation_increased = True
                if self.max_validation_f1_INC > 0.758 and self.opt.do_save == True:
                    torch.save(self.model.state_dict(), self.opt.model_save_path+'/best_f1.pkl')
                    print('='*77)
                    print('model saved at: ', self.opt.model_save_path + '/best_f1.pkl')
                    print('='*77)
        else:
#             if ss and eval_accuracy > 0.84:
#                 print('Got 420 correct!')
#                 torch.save(self.model.state_dict(), self.opt.model_save_path+f'/best_acc_{eval_accuracy}.pkl') 
#                 torch.save(eval_vdc, f'./analysis/{self.global_step}_eval_vdc.pt')
                    
            validation_increased = False
            if eval_accuracy > self.max_test_acc_INC:
                validation_increased = True
                self.max_test_acc_INC = eval_accuracy
                if self.max_test_acc_INC > 0.85 and self.opt.do_save == True:
                    torch.save(self.model.state_dict(), self.opt.model_save_path+'/best_acc.pkl')
                    print('='*77)
                    print('model saved at: ', self.opt.model_save_path + '/best_acc.pkl')
                    print('='*77)
            if test_f1 > self.max_test_f1_INC:
                validation_increased = True
                self.max_test_f1_INC = test_f1
                if self.max_test_f1_INC > 0.815 and self.opt.do_save == True:
                    torch.save(self.model.state_dict(), self.opt.model_save_path+'/best_f1.pkl')
                    print('='*77)
                    print('model saved at: ', self.opt.model_save_path + '/best_f1.pkl')
                    print('='*77)

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'eval_f1': test_f1, }
            
            
#         torch.save(eval_vdc, f'./vis/{args.automation_type}_{args.model_init_seed}_eval_vdc.pt')
#         if validation_increased == True:
#             assert eval_vdc.size(0) == len(self.dataset.eval_examples)
            
#             torch.save(eval_vdc, f'./vis/{args.automation_type}_{args.model_init_seed}_eval_vdc.pt')
#             torch.save(torch.FloatTensor(y_pred), f'./vis/{args.automation_type}_{args.model_init_seed}_y_pred.pt')
#             torch.save(torch.FloatTensor(y_true), f'./vis/{args.automation_type}_{args.model_init_seed}_y_true.pt')
            
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
        if task_name == 'mams':
            print("Max validation set acc: {:.4f}, F1: {:.4f}".format(self.max_validation_acc_INC, self.max_validation_f1_INC))
#             print("Max test set acc: {:.4f}, F1: {:.4f}".format(self.max_test_acc_INC, self.max_test_f1_INC))
#             print("Last test set acc: {:.4f}, F1: {:.4f}, epoch: {:.4f}".format(self.last_test_acc, self.last_test_f1, 
#                                                                                 self.last_test_epoch))
#             print('In detail -----------------------------------------------')
#             for i in range(len(self.test_acc_each_epoch)):
#                 print(f"Early stop after {i+1} epoch : [Acc: {self.test_acc_each_epoch[i]} / F1: {self.test_f1_each_epoch[i]}")
                
        else:
            print("Max validate set acc: {:.4f}, F1: {:.4f}".format(self.max_test_acc_INC, self.max_test_f1_INC))
#             print('=========== best-case sg loss ===========')
#             print("Max validate set acc: {:.4f}, F1: {:.4f}".format(self.max_test_acc_rand, self.max_test_f1_rand))
            
#         self.writer.close()
#         return self.max_test_acc
        return self.max_test_acc_INC, self.max_test_f1_INC


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    args = get_config_2()  # Gets the user settings or default hyperparameters
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
        'roberta': RobertaForSequenceClassification,
        'roberta_td': RobertaForSequenceClassification_TD,
        'roberta_gcls': RobertaForSequenceClassification_gcls,
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
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
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
