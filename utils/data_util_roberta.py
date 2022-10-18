import tokenization_word_roberta as tokenization
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from data_utils import *
from bucket_iterator import BucketIterator
from bucket_iterator_2_roberta import BucketIterator_2
import pickle
from transformers import BertTokenizer, RobertaTokenizer

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def length2mask(length,maxlength):
    size=list(length.size())
    length=length.unsqueeze(-1).repeat(*([1]*len(size)+[maxlength])).long()
    ran=torch.arange(maxlength)
    ran=ran.expand_as(length)
    mask=ran<length
    return mask.float()

# Read the dataset and preprocess it
class ReadData:
    def __init__(self, opt):
        print("load data ...")
        self.opt = opt
        self.train_examples = opt.processor.get_train_examples(opt.data_dir)
        self.eval_examples = opt.processor.get_dev_examples(opt.data_dir)
        if opt.task_name == 'mams':
            self.eval_examples_2 = opt.processor.get_test_examples(opt.data_dir)
        self.label_list = opt.processor.get_labels()
        
        self.tokenizer = tokenization.FullTokenizer(vocab_file=opt.vocab_file, do_lower_case=opt.do_lower_case)
        ######################
        print('-'*100)
        print("Combining with dataloader from DGEDT (e.g. elements like dependency graphs are needed).")
        print('-'*100)
        if opt.task_name == 'laptop':
            dgedt_dataset = 'lap14'
        elif opt.task_name == 'restaurant':
            dgedt_dataset = 'rest14'
        elif opt.task_name == 'tweet':
            dgedt_dataset = 'twitter'
        elif opt.task_name == 'mams':
            dgedt_dataset = 'mams'
            
        absa_dataset=pickle.load(open(dgedt_dataset+'_datas_roberta.pkl', 'rb'))
        opt.edge_size=len(absa_dataset.edgevocab)
        self.train_data_loader = BucketIterator_2(data=absa_dataset.train_data, batch_size=100000, max_seq_length = self.opt.max_seq_length, shuffle=True)
        self.test_data_loader = BucketIterator_2(data=absa_dataset.test_data, batch_size=100000, max_seq_length = self.opt.max_seq_length, shuffle=False)
        
        self.DGEDT_train_data = self.train_data_loader.data
        self.DGEDT_train_batches = self.train_data_loader.batches
        
        self.DGEDT_test_data = self.test_data_loader.data
        self.DGEDT_test_batches = self.test_data_loader.batches
        
        if opt.model_name in ['gcls', 'scls', 'roberta', 'gcls_er', 'gcls_moe', 'roberta_gcls', 'roberta_gcls_2', 'roberta_gcls_moe', 'roberta_td', 'roberta_lcf_td', 'roberta_gcls_td', 'roberta_asc_td']:
            
            if opt.graph_type == 'dg':
                self.train_gcls_attention_mask = self.process_DG(self.DGEDT_train_data, self.opt.L_config_base, 
                                                                 path_types = self.opt.path_types)
                self.eval_gcls_attention_mask = self.process_DG(self.DGEDT_test_data, self.opt.L_config_base, 
                                                                 path_types = self.opt.path_types)
                
            elif opt.graph_type == 'sd':
                self.train_gcls_attention_mask  = self.process_surface_distance(self.DGEDT_train_data, self.opt.L_config_base)
                self.eval_gcls_attention_mask = self.process_surface_distance(self.DGEDT_test_data, self.opt.L_config_base)
            
            
        ######################
        
        self.train_data, self.train_dataloader, self.train_tran_indices, self.train_span_indices, \
        self.train_extended_attention_mask  = self.get_data_loader(examples=self.train_examples, type='train_data',
                                                                   gcls_attention_mask = self.train_gcls_attention_mask, 
                                                                   path_types = self.opt.path_types)
        self.eval_data, self.eval_dataloader, self.eval_tran_indices, self.eval_span_indices, \
        self.eval_extended_attention_mask = self.get_data_loader(examples=self.eval_examples, type='eval_data', 
                                                                 gcls_attention_mask = self.eval_gcls_attention_mask,
                                                                 path_types = self.opt.path_types)
                  
    def process_DG(self, DGEDT_train_data, layer_L, path_types = None):
        length_R_trans = {}
        cumul_R_trans = {}
        total_tokens = [] # Used for doing DG analysis
        
        gcls_attention_mask = torch.zeros((len(DGEDT_train_data), max(layer_L)+1, path_types, 128), dtype=torch.float)
        for i in range(len(DGEDT_train_data)):
            length_R_trans[i] = {}
            cumul_R_trans[i] = {}
            for j in range(max(layer_L)+1):
                length_R_trans[i][j] = {}
                cumul_R_trans[i][j] = {}
                for k in range(min(2**j, path_types)):
                    length_R_trans[i][j][k] = []
                    cumul_R_trans[i][j][k] = []
                    
        target_idx = []
        
        for i in range(len(DGEDT_train_data)):
            dg = DGEDT_train_data[i]['dependency_graph'][0] + DGEDT_train_data[i]['dependency_graph'][1]
            dg[dg>=1] = 1
            dg_in = DGEDT_train_data[i]['dependency_graph'][0]
            dg_out = DGEDT_train_data[i]['dependency_graph'][1]
            
            dg = torch.tensor(dg)
            dg_in = torch.tensor(dg_in)
            dg_out = torch.tensor(dg_out)
            
            sep_pos = len(DGEDT_train_data[i]['text_indices']) 
            total_tokens.append(sep_pos-2)
            
            if len(DGEDT_train_data[i]['span_indices']) != 1:
                print('Multiple same aspects in the sentence : ', i)
#             assert len(DGEDT_train_data[i]['span_indices']) == 1    # At least for Laptop and Restaurant.
            
            length_0_trans = []
            for k in range(len(DGEDT_train_data[i]['span_indices'])):
                tran_start = DGEDT_train_data[i]['span_indices'][k][0]
                tran_end = DGEDT_train_data[i]['span_indices'][k][1]

                for item in range(tran_start, tran_end):
                    length_R_trans[i][0][0].append([item])
                    cumul_R_trans[i][0][0].append([item])
                    length_0_trans.append(item)
            
            used_trans = []
            for l in range(1, max(layer_L)+1):
                #######
                for j in range(min(2**(l-1), path_types)):
                    if 2**l <= path_types:
                        for k in range(2):
                            if k == 0:
                                dg_ = dg_in
                            else:
                                dg_ = dg_out
                            
                            cumul_R_trans[i][l][2*j+k] = cumul_R_trans[i][l-1][j].copy()
                            
                            for path in length_R_trans[i][l-1][j]:
                                last_node = path[-1]
                                if len(path) > 1:
                                    prev_node = path[-2]
                                else:
                                    prev_node = None

                                x = (dg_[last_node] == 1).nonzero(as_tuple=True)[0]
                                
                                for item in x:
                                    if int(item) not in length_0_trans:
                                        if path_types == 1:    # path_types = 1 인 경우는 언제나 최단 경로를 선택하기 위함.
                                            if int(item) in used_trans:
                                                continue
                                        if int(item) != last_node and int(item) != prev_node:
                                                length_R_trans[i][l][2*j+k].append(path+[int(item)])
                                                cumul_R_trans[i][l][2*j+k].append(path+[int(item)])
                                                used_trans.append(int(item))
                    else:
                        dg_ = dg
                        cumul_R_trans[i][l][j] = cumul_R_trans[i][l-1][j].copy()
                        for path in length_R_trans[i][l-1][j]:
                            last_node = path[-1]
                            if len(path) > 1:
                                prev_node = path[-2]
                            else:
                                prev_node = None

                            x = (dg_[last_node] == 1).nonzero(as_tuple=True)[0]
                            for item in x:
                                if int(item) not in length_0_trans:
                                    if path_types == 1:
                                        if int(item) in used_trans:
                                            continue
                                    if int(item) != last_node and int(item) != prev_node:
                                            length_R_trans[i][l][j].append(path+[int(item)])
                                            cumul_R_trans[i][l][j].append(path+[int(item)])
                                            used_trans.append(int(item))
                                            
            
            ## 점검
            for j in range(len(length_R_trans[i])):
                for k in range(len(length_R_trans[i][j])):
                    for item in length_R_trans[i][j][k]:
                        assert len(item) == j+1
            
            # Converting to gcls_att_mask. 
            aspect_length = len(DGEDT_train_data[i]['aspect_indices']) - 2
            if self.opt.model_name in ['roberta_gcls']:
                A =2 

#                 A = 1    # for using target at the end as Graph-s.
#                 available_room = 128 - len([0] + DGEDT_train_data[i]['text_indices'][1:] + [2] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize('target is')) + DGEDT_train_data[i]['aspect_indices']) -1 # -1 for the comma.
#                 if available_room > aspect_length:
#                     A = 1 + aspect_length + 1 # +1 for the comma
#                 else:
#                     A = 1 + available_room
            elif self.opt.model_name in ['roberta_gcls_2']:
                A = 3
            elif self.opt.model_name in ['roberta_td', 'roberta']:
                A = 1
            
            # target_idx
            for item in length_R_trans[i][0][0]:
                assert len(item) == 1
                target_idx.append(DGEDT_train_data[i]['tran_indices'][item[0]][0] + A)
                break
            
            # Construct gcls_attention_mask
            for j in range(len(length_R_trans[i])):
                for k in range(len(length_R_trans[i][j])):
                    att_mask = torch.zeros([128])
                    for item in cumul_R_trans[i][j][k]:
                        start_idx = DGEDT_train_data[i]['tran_indices'][item[-1]][0] + A
                        end_idx = DGEDT_train_data[i]['tran_indices'][item[-1]][1] + A
                        
                        att_mask[start_idx:end_idx] = 1
                        
                    gcls_attention_mask[i][j][k] = att_mask

#         overlap doesn't occur when cumul = False. Overlap inevitabily occurs for cumul = True
#         print('='*77)
#         print('Checking any overlaps')
#         for i in range(len(DGEDT_train_data)):
#             for j in range(max(layer_L)+1):
#                 for k in range(j+1, max(layer_L)+1):
#                     x = gcls_attention_mask[i][j][0] + gcls_attention_mask[i][k][0]
#                     y = (x == 2).nonzero(as_tuple=True)[0]
#                     if len(y) != 0:
#                         print('='*77)
#                         print('i: ', i)
#                         print('j: ', j)
#                         print('k: ', k)
#                         print('y: ', y)
#                         print('length_R_trans[i][0][0]: ', length_R_trans[i][0][0]) 
#                         print('gcls_attention_mask[i][j][0]: ', gcls_attention_mask[i][j][0])
#                         print('gcls_attention_mask[i][k][0]: ', gcls_attention_mask[i][k][0])


        print('='*77)
        print('reporting DG statistics')
        total_toks = 0
        R_tokens = torch.zeros((max(layer_L)+1, path_types), dtype = torch.float)
        
        for i in range(len(DGEDT_train_data)):
            total_toks += total_tokens[i]
            for j in range(max(layer_L)+1):
                for k in range(path_types):
                    R_tokens[j,k] += torch.sum(gcls_attention_mask[i][j][k])
                    
        for i in range(max(layer_L)+1):
            print('-'*77)
            print('Range: ', i)
            for j in range(path_types):
                print(f'portion of tokens in range {i} / path type {j}: ', R_tokens[i][j] / total_toks)
                
        return gcls_attention_mask

    def process_surface_distance(self, DGEDT_train_data, layer_L):
        final_all_paths = []
        length_0_trans = []
        target_idx = []
        
        gcls_attention_mask = torch.empty((len(DGEDT_train_data), max(layer_L)+1, 128), dtype=torch.float)

        for i in range(len(DGEDT_train_data)):
            all_paths = []
            length_0_trans.append([])

#             assert len(DGEDT_train_data[i]['span_indices']) == 1    # At least for Laptop and Restaurant.
            for j in range(len(DGEDT_train_data[i]['span_indices'])):
                tran_start = DGEDT_train_data[i]['span_indices'][j][0]
                tran_end = DGEDT_train_data[i]['span_indices'][j][1]

                for item in range(tran_start, tran_end):
                    all_paths.append([item])
                    length_0_trans[i].append(item)

#             length_0_trans[i] = [item for item in range(tran_start, tran_end)]

            final_all_paths.append(all_paths)
            
            # Now let's make the gcls_att_mask.
            if self.opt.model_name in ['roberta_gcls']:
                A = 2
            elif self.opt.model_name in ['roberta_gcls_2']:
                A = 3
            elif self.opt.model_name in ['roberta', 'roberta_td']:
                A = 1
            
            sep_pos = len(DGEDT_train_data[i]['text_indices'])
            for j in range(max(layer_L)+1):
                att_mask = torch.zeros([128])
                for item_ in length_0_trans[i]: 
                    start_idx = DGEDT_train_data[i]['tran_indices'][item_][0] + A - j
                    end_idx = DGEDT_train_data[i]['tran_indices'][item_][1] + A + j
                
                    if start_idx >= A:
                        att_mask[start_idx:end_idx] = 1
                    else:
                        att_mask[A:end_idx] = 1
                    att_mask[sep_pos:128] = 0
                            
                gcls_attention_mask[i][j] = att_mask

#             if torch.sum(gcls_attention_mask[i][1]) - torch.sum(gcls_attention_mask[i][0]) != 2:
#                 print('='*77)
#                 print(i)
#                 print(tokenizer.convert_ids_to_tokens(DGEDT_train_data[i]['text_indices']))
#                 print(tokenizer.convert_ids_to_tokens(DGEDT_train_data[i]['aspect_indices']))
#                 print('='*77)
#             if torch.sum(gcls_attention_mask[i][2]) - torch.sum(gcls_attention_mask[i][1]) != 2:
#                 print(i)
#                 print(tokenizer.convert_ids_to_tokens(DGEDT_train_data[i]['text_indices']))
#                 print(tokenizer.convert_ids_to_tokens(DGEDT_train_data[i]['aspect_indices']))

        return gcls_attention_mask
            
    ###### For implementing RoBERTa-LCF and RoBERTa-LCFS 
    def get_cdw_vec(self, local_indices, aspect_indices, aspect_begin, syntactical_dist=None):
        SRD = 1.0
        cdw_vec = np.zeros((self.opt.max_seq_length), dtype=np.float32)
        
        text_len = int((local_indices == 2).nonzero(as_tuple=True)[0][0] +1)
        aspect_len = int((aspect_indices == 2).nonzero(as_tuple=True)[0][0] - 1)
        
        if syntactical_dist is not None:
            for i in range(min(text_len, self.opt.max_seq_length)):
                if syntactical_dist[i] > SRD:
                    w = 1 - syntactical_dist[i] / text_len
                    cdw_vec[i] = w
                else:
                    cdw_vec[i] = 1
        else:
            local_context_begin = max(0, aspect_begin - SRD)
            local_context_end = min(aspect_begin + aspect_len + SRD - 1, self.opt.max_seq_length)
            for i in range(min(text_len, self.opt.max_seq_length)):
                if i < local_context_begin:
                    w = 1 - (local_context_begin - i) / text_len
                elif local_context_begin <= i <= local_context_end:
                    w = 1
                else:
                    w = 1 - (i - local_context_end) / text_len
                try:
                    assert 0 <= w <= 1  # exception
                except:
                    pass
                    # print('Warning! invalid CDW weight:', w)
                cdw_vec[i] = w

#         cdw_vec[0] = 1

        return cdw_vec

    def get_data_loader(self, examples, type='train_data', gcls_attention_mask=None, path_types = None):
        features = self.convert_examples_to_features(
            examples, self.label_list, self.opt.max_seq_length, self.tokenizer)
        
        if type == 'train_data':
            DGEDT_batches = self.DGEDT_train_batches[0]
            DGEDT_data = self.DGEDT_train_data    # 점검용
        elif type == 'eval_data':
            DGEDT_batches = self.DGEDT_test_batches[0]
            DGEDT_data = self.DGEDT_test_data    # 점검용
        
        print(type)
        print('='*77)
        batch_size_ = DGEDT_batches['text_indices'].size(0)
        
        all_input_ids_org = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        
        assert all_input_ids_org.size(0) == batch_size_
        ##############################
        all_input_ids = DGEDT_batches['text_indices']
        all_input_ids_lcf_global = DGEDT_batches['text_indices_lcf_global']
        all_input_ids_lcf_local = DGEDT_batches['text_indices_lcf_local']
        
        
        ##############################
        
#         all_graph_s_pos = DGEDT_batches['graph_s_pos']
        
        all_input_mask_org = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        ##############################
        text_len = torch.sum(DGEDT_batches['text_indices'] != 1, dim=-1)
        text_len_lcf_global = torch.sum(DGEDT_batches['text_indices_lcf_global'] != 1, dim=-1)
        text_len_lcf_local = torch.sum(DGEDT_batches['text_indices_lcf_local'] != 1, dim=-1)
        
        
        all_input_mask = length2mask(text_len, DGEDT_batches['text_indices'].size(1))
        all_input_mask_lcf_global = length2mask(text_len, DGEDT_batches['text_indices_lcf_global'].size(1))
        all_input_mask_lcf_local = length2mask(text_len, DGEDT_batches['text_indices_lcf_local'].size(1))
        
        all_segment_ids_org = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        ##############################
        all_segment_ids = all_segment_ids_org  # all zeroes of size 128.
        all_segment_ids_lcf_global = all_segment_ids_org
        all_segment_ids_lcf_local = all_segment_ids_org
        
        ########## When target is appended at the end or the beginning.
        for i in range(batch_size_):
            x = (all_input_ids[i] == 2).nonzero(as_tuple=True)[0]
            all_segment_ids[i][x[0]+1:x[-1]+1] = 1
            
            x = (all_input_ids_lcf_global[i] == 2).nonzero(as_tuple=True)[0]
            all_segment_ids_lcf_global[i][x[0]+1:x[-1]+1] = 1
            
            x = (all_input_ids_lcf_local[i] == 2).nonzero(as_tuple=True)[0]
            all_segment_ids_lcf_global[i][x[0]+1:x[-1]+1] = 1
            # lcf_local -> 전부 0.
            
        ##########
    
        all_label_ids_org = torch.tensor([f.label_id for f in features], dtype=torch.long)
        ##############################
        all_label_ids = all_label_ids_org
        for i in range(batch_size_):
            assert all_label_ids[i] == DGEDT_batches['polarity'][i]
        # Note that DGEDT, TD-BERT guid 순서는 동일.
        # No change. Quite obviously, the labels are the same.
        ##############################
        
        
                
        all_input_t_ids_org = torch.tensor([f.input_t_ids for f in features], dtype=torch.long)
        ##############################
        all_input_t_ids = DGEDT_batches['aspect_indices']
        ##############################
        
        
        
        all_input_t_mask_org = torch.tensor([f.input_t_mask for f in features], dtype=torch.long)
        ##############################
        all_input_t_mask = torch.tensor(all_input_t_ids>0, dtype = torch.long)
        # Not used anyway
        ##############################

        
        
        all_segment_t_ids_org = torch.tensor([f.segment_t_ids for f in features], dtype=torch.long)
        ##############################
        all_segment_t_ids = torch.zeros(all_input_t_ids.size(), dtype = torch.long)
        # No change. Zeros with size 128. Also not used anyway.
        ##############################

        
        
        all_input_without_t_ids_org = torch.tensor([f.input_without_t_ids for f in features], dtype=torch.long)
        ##############################
        # not used in TD-BERT.
        all_input_without_t_ids = torch.zeros(all_input_without_t_ids_org.size(), dtype = torch.long)
        ##############################
        
        
        
        all_input_without_t_mask_org = torch.tensor([f.input_without_t_mask for f in features], dtype=torch.long)
        ##############################
        # not used in TD-BERT.
        all_input_without_t_mask = torch.zeros(all_input_without_t_mask_org.size(), dtype = torch.long)
        ##############################
        
        
        
        all_segment_without_t_ids_org = torch.tensor([f.segment_without_t_ids for f in features], dtype=torch.long)
        ##############################
        # not used in TD-BERT.
        all_segment_without_t_ids = torch.zeros(all_segment_without_t_ids_org.size(), dtype = torch.long)
        ##############################
        
        
        
        all_input_left_t_ids = torch.tensor([f.input_left_t_ids for f in features], dtype=torch.long)
        all_input_left_t_mask = torch.tensor([f.input_left_t_mask for f in features], dtype=torch.long)
        all_segment_left_t_ids = torch.tensor([f.segment_left_t_ids for f in features], dtype=torch.long)
        all_input_right_t_ids = torch.tensor([f.input_right_t_ids for f in features], dtype=torch.long)
        all_input_right_t_mask = torch.tensor([f.input_right_t_mask for f in features], dtype=torch.long)
        all_segment_right_t_ids = torch.tensor([f.segment_right_t_ids for f in features], dtype=torch.long)
        ##############################
        # Above six are all not used in TD-BERT.
        ##############################

        input_left_ids_org = torch.tensor([f.input_left_ids for f in features], dtype=torch.long)
        ##############################
        input_left_ids = torch.zeros(input_left_ids_org.size(), dtype = torch.long)
        for i in range(batch_size_):
            aspect_start_idx = DGEDT_batches['tran_indices'][i][DGEDT_batches['span_indices'][i][0][0]][0] + 1
            input_left_ids[i][:aspect_start_idx] = DGEDT_batches['text_indices'][i][:aspect_start_idx]
            input_left_ids[i][aspect_start_idx] = 2    # [SEP]
            
        ##############################
        
        
        
        input_left_mask_org = torch.tensor([f.input_left_mask for f in features], dtype=torch.long)
        ##############################
        input_left_mask = torch.tensor(input_left_ids>0, dtype = torch.long)
        # 어차피 안 쓰임.
        ##############################
        
        
        
        segment_left_ids_org = torch.tensor([f.segment_left_ids for f in features], dtype=torch.long)
        ##############################
        segment_left_ids = torch.zeros(input_left_ids.size(), dtype = torch.long)
        # 어차피 안 쓰임.
        ##############################
        
        all_tran_indices = DGEDT_batches['tran_indices']
        all_span_indices = DGEDT_batches['span_indices']
        all_input_guids = torch.tensor([i for i in range(batch_size_)], dtype = torch.long)
        all_input_dg = DGEDT_batches['dependency_graph']
        all_input_dg1 = DGEDT_batches['dependency_graph1']
        all_input_dg2 = DGEDT_batches['dependency_graph2']
        all_input_dg3 = DGEDT_batches['dependency_graph3']
        
        ###### Making LCF matrices
        lcf_vec_list = []
        for j in range(batch_size_):
            aspect_start_idx = DGEDT_batches['tran_indices'][j][DGEDT_batches['span_indices'][j][0][0]][0] + 1
            lcf_vec_list.append(self.get_cdw_vec(local_indices=all_input_ids_lcf_local[j], 
                                                 aspect_indices=all_input_t_ids[j], aspect_begin=aspect_start_idx,
                                                 syntactical_dist=None))

#         lcf_vec = torch.tensor(lcf_vec_list, dtype=torch.long)


        ###############################
        extended_attention_mask = torch.zeros([all_input_ids.size(0),1,1,128])    # torch.Size([32, 1, 1, 128])
        print('extended_attention_mask.size() should be [N,1,1,128] ', extended_attention_mask.size())
        extended_attention_mask = extended_attention_mask.repeat(1,1,all_input_ids.size(1),1)    # torch.Size([32, 1, 128, 128])
        print('extended_attention_mask.size() should be [N,1,128,128] ', extended_attention_mask.size())

#         extended_attention_mask = extended_attention_mask.unsqueeze(1).repeat(1,12,1,1,1)
#         print('extended_attention_mask.size() should be [N,12,1,128,128] ', extended_attention_mask.size())

        # head 고려
        extended_attention_mask = extended_attention_mask.unsqueeze(1).repeat(1,12,1,1,1)
        print('extended_attention_mask.size() should be [N,12,12,128,128] ', extended_attention_mask.size())

#         for j, item in enumerate(self.opt.L_config_base):
#             for i in range(all_input_ids.size(0)):
#                 extended_attention_mask[i, j, 0, 1, :] =  (1 - gcls_attention_mask[i][item]) * -10000.0

        g_config = []
        if len(self.opt.g_config) == 4:
            for i in range(12):
                g_config.append(torch.tensor([[self.opt.g_config[0], self.opt.g_config[1]],
                                              [self.opt.g_config[2], self.opt.g_config[3]]], dtype = torch.float))
        
        elif len(self.opt.g_config) == 9:
            for i in range(12):
                if i < self.opt.g_config[8]:
                    g_config.append(torch.tensor([[self.opt.g_config[0], self.opt.g_config[1]],
                                                  [self.opt.g_config[2], self.opt.g_config[3]]], dtype = torch.float))
                else:
                    g_config.append(torch.tensor([[self.opt.g_config[4], self.opt.g_config[5]],
                                                  [self.opt.g_config[6], self.opt.g_config[7]]], dtype = torch.float))
        
        for i in range(all_input_ids.size(0)):
            for j, item in enumerate(self.opt.L_config_base):
#                 extended_attention_mask[i, j, 0, 1, :] =  (1 - gcls_attention_mask[i][item]) * -10000.0

#                 extended_attention_mask[i, j, 0, 0, 0] = (1-g_config[j][0][0]) * -10000.0
#                 extended_attention_mask[i, j, 0, 0, self.opt.gcls_pos] = (1-g_config[j][0][1]) * -10000.0
#                 extended_attention_mask[i, j, 0, self.opt.gcls_pos, 0] = (1-g_config[j][1][0]) * -10000.0
#                 extended_attention_mask[i, j, 0, self.opt.gcls_pos, self.opt.gcls_pos] = (1-g_config[j][1][1]) * -10000.0

                if self.opt.graph_type == 'dg':
                    extended_attention_mask[i, j, 0, 1, :] =  (1 - gcls_attention_mask[i][item][0]) * -10000.0
                elif self.opt.graph_type == 'sd':
                    extended_attention_mask[i, j, 0, 1, :] =  (1 - gcls_attention_mask[i][item]) * -10000.0


#                 if item == 2 and self.opt.path_types >= 4:
#                     extended_attention_mask[i, j, 0, 1, :] =  (1 - gcls_attention_mask[i][item][0]) * -10000.0
#                     extended_attention_mask[i, j, 3, 1, :] =  (1 - gcls_attention_mask[i][item][0]) * -10000.0
#                     extended_attention_mask[i, j, 6, 1, :] =  (1 - gcls_attention_mask[i][item][0]) * -10000.0
#                     extended_attention_mask[i, j, 9, 1, :] =  (1 - gcls_attention_mask[i][item][0]) * -10000.0

#                     extended_attention_mask[i, j, 1, 1, :] =  (1 - gcls_attention_mask[i][item][2]) * -10000.0
#                     extended_attention_mask[i, j, 4, 1, :] =  (1 - gcls_attention_mask[i][item][2]) * -10000.0
#                     extended_attention_mask[i, j, 7, 1, :] =  (1 - gcls_attention_mask[i][item][2]) * -10000.0
#                     extended_attention_mask[i, j, 10, 1, :] =  (1 - gcls_attention_mask[i][item][2]) * -10000.0

#                     extended_attention_mask[i, j, 2, 1, :] =  (1 - gcls_attention_mask[i][item][3]) * -10000.0
#                     extended_attention_mask[i, j, 5, 1, :] =  (1 - gcls_attention_mask[i][item][3]) * -10000.0
#                     extended_attention_mask[i, j, 8, 1, :] =  (1 - gcls_attention_mask[i][item][3]) * -10000.0
#                     extended_attention_mask[i, j, 11, 1, :] =  (1 - gcls_attention_mask[i][item][3]) * -10000.0

#                 elif item == 3 and self.opt.path_types >= 8:
#                     extended_attention_mask[i, j, 0, 1, :] =  (1 - gcls_attention_mask[i][item][0]) * -10000.0
#                     extended_attention_mask[i, j, 4, 1, :] =  (1 - gcls_attention_mask[i][item][0]) * -10000.0
#                     extended_attention_mask[i, j, 8, 1, :] =  (1 - gcls_attention_mask[i][item][0]) * -10000.0

#                     extended_attention_mask[i, j, 1, 1, :] =  (1 - gcls_attention_mask[i][item][4]) * -10000.0
#                     extended_attention_mask[i, j, 5, 1, :] =  (1 - gcls_attention_mask[i][item][4]) * -10000.0
#                     extended_attention_mask[i, j, 9, 1, :] =  (1 - gcls_attention_mask[i][item][4]) * -10000.0

#                     extended_attention_mask[i, j, 2, 1, :] =  (1 - gcls_attention_mask[i][item][6]) * -10000.0
#                     extended_attention_mask[i, j, 6, 1, :] =  (1 - gcls_attention_mask[i][item][6]) * -10000.0
#                     extended_attention_mask[i, j, 10, 1, :] =  (1 - gcls_attention_mask[i][item][6]) * -10000.0

#                     extended_attention_mask[i, j, 3, 1, :] =  (1 - gcls_attention_mask[i][item][7]) * -10000.0
#                     extended_attention_mask[i, j, 7, 1, :] =  (1 - gcls_attention_mask[i][item][7]) * -10000.0
#                     extended_attention_mask[i, j, 11, 1, :] =  (1 - gcls_attention_mask[i][item][7]) * -10000.0

#                 else:
#                     for t in range(min(2**item, path_types)):
#                         extended_attention_mask[i, j, (int(12/min(2**item, path_types)))*t:int((12/min(2**item, path_types)))*(t+1), 1, :] =  (1 - gcls_attention_mask[i][item][t]) * -10000.0

                extended_attention_mask[i, j, :, 0, 0] = (1-g_config[j][0][0]) * -10000.0
                extended_attention_mask[i, j, :, 0, self.opt.g_token_pos] = (1-g_config[j][0][1]) * -10000.0
                extended_attention_mask[i, j, :, self.opt.g_token_pos, 0] = (1-g_config[j][1][0]) * -10000.0
                extended_attention_mask[i, j, :, self.opt.g_token_pos, self.opt.g_token_pos] = (1-g_config[j][1][1]) * -10000.0
                
                
        ###############################

#         data = TensorDataset(all_input_ids, all_graph_s_pos, all_input_ids_lcf_global, all_input_ids_lcf_local, all_input_mask, 
#                              all_input_mask_lcf_global, all_input_mask_lcf_local, all_segment_ids, 
#                              all_segment_ids_lcf_global, all_segment_ids_lcf_local, all_label_ids, all_input_t_ids,
#                              all_input_t_mask, all_segment_t_ids, all_input_without_t_ids, 
#                              all_input_without_t_mask, all_segment_without_t_ids, all_input_left_t_ids, all_input_left_t_mask,
#                              all_segment_left_t_ids,all_input_right_t_ids, all_input_right_t_mask, all_segment_right_t_ids,
#                              input_left_ids, input_left_mask, segment_left_ids, all_input_dg, all_input_dg1, 
#                              all_input_dg2, all_input_dg3, all_input_guids, gcls_attention_mask)

        data = TensorDataset(all_input_ids, all_label_ids, all_input_guids)
        
        if type == 'train_data':
            train_data = data
            train_sampler = RandomSampler(data)
            return train_data, DataLoader(train_data, sampler=train_sampler, batch_size=self.opt.train_batch_size), all_tran_indices, all_span_indices, extended_attention_mask
        else:
            eval_data = data
            eval_sampler = SequentialSampler(eval_data)
            return eval_data, DataLoader(eval_data, sampler=eval_sampler, batch_size=self.opt.eval_batch_size), all_tran_indices, all_span_indices, extended_attention_mask

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)
            
            if len(tokens_a) >= 100-2:    # this might have error since DGEDT's tokenization and TD-BERT's tokenization is not the same. But of course, very less likely to have an error.
                continue
            tokens_aspect = tokenizer.tokenize(example.aspect)
            tokens_text_without_target = tokenizer.tokenize(example.text_without_target)
            tokens_text_left_with_target = tokenizer.tokenize(example.text_left_with_target)
            tokens_text_right_with_target = tokenizer.tokenize(example.text_right_with_target)
            tokens_text_left = tokenizer.tokenize(example.text_left)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[0:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            segment_ids = []
            tokens.append("<s>")
            segment_ids.append(0)
            for token in tokens_a:
                if token == '[UNK]':
                    token = 'unk'
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("</s>")
            segment_ids.append(0)

            if tokens_aspect:  # if not None
                tokens_t = []
                segment_t_ids = []
                tokens_t.append("<s>")
                segment_t_ids.append(0)
                for token in tokens_aspect:
                    if token == '[UNK]':
                        token = 'unk'
                    tokens_t.append(token)
                    segment_t_ids.append(0)
                tokens_t.append("</s>")
                segment_t_ids.append(0)
                input_t_ids = tokenizer.convert_tokens_to_ids(tokens_t)
                input_t_mask = [1] * len(input_t_ids)
                while len(input_t_ids) < max_seq_length:
                    input_t_ids.append(0)
                    input_t_mask.append(0)
                    segment_t_ids.append(0)
                assert len(input_t_ids) == max_seq_length
                assert len(input_t_mask) == max_seq_length
                assert len(segment_t_ids) == max_seq_length
                # The following is the case where the target word is removed from the processing sentence, tokens_text_without_target
                tokens_without_target = []
                segment_without_t_ids = []
                tokens_without_target.append("<s>")
                segment_without_t_ids.append(0)
                for token in tokens_text_without_target:
                    if token == '[UNK]':
                        token = 'unk'
                    tokens_without_target.append(token)
                    segment_without_t_ids.append(0)
                tokens_without_target.append("</s>")
                segment_without_t_ids.append(0)
                input_without_t_ids = tokenizer.convert_tokens_to_ids(tokens_without_target)
                input_without_t_mask = [1] * len(input_without_t_ids)
                while len(input_without_t_ids) < max_seq_length:
                    input_without_t_ids.append(0)
                    input_without_t_mask.append(0)
                    segment_without_t_ids.append(0)
                assert len(input_without_t_ids) == max_seq_length
                assert len(input_without_t_mask) == max_seq_length
                assert len(segment_without_t_ids) == max_seq_length
                # The following is the sentence processing the left side of the target word, containing the target word, tokens_text_left_with_aspect
                tokens_left_target = []
                segment_left_t_ids = []
                tokens_left_target.append("<s>")
                segment_left_t_ids.append(0)
                for token in tokens_text_left_with_target:
                    if token == '[UNK]':
                        token = 'unk'
                    tokens_left_target.append(token)
                    segment_left_t_ids.append(0)
                tokens_left_target.append("</s>")
                segment_left_t_ids.append(0)
                input_left_t_ids = tokenizer.convert_tokens_to_ids(tokens_left_target)
                input_left_t_mask = [1] * len(input_left_t_ids)
                while len(input_left_t_ids) < max_seq_length:
                    input_left_t_ids.append(0)
                    input_left_t_mask.append(0)
                    segment_left_t_ids.append(0)
                assert len(input_left_t_ids) == max_seq_length
                assert len(input_left_t_mask) == max_seq_length
                assert len(segment_left_t_ids) == max_seq_length
                # The following is the sentence processing the right side of the target word, containing the target word, tokens_text_right_with_aspect
                tokens_right_target = []
                segment_right_t_ids = []
                tokens_right_target.append("<s>")
                segment_right_t_ids.append(0)
                for token in tokens_text_right_with_target:
                    if token == '[UNK]':
                        token = 'unk'
                    tokens_right_target.append(token)
                    segment_right_t_ids.append(0)
                tokens_right_target.append("</s>")
                segment_right_t_ids.append(0)
                input_right_t_ids = tokenizer.convert_tokens_to_ids(tokens_right_target)
                input_right_t_mask = [1] * len(input_right_t_ids)
                while len(input_right_t_ids) < max_seq_length:
                    input_right_t_ids.append(0)
                    input_right_t_mask.append(0)
                    segment_right_t_ids.append(0)
                assert len(input_right_t_ids) == max_seq_length
                assert len(input_right_t_mask) == max_seq_length
                assert len(segment_right_t_ids) == max_seq_length
                # The following are sentences that process the left side of the target word and do not contain the target word, tokens_text_left
                tokens_left = []
                segment_left_ids = []
                tokens_left.append("<s>")
                segment_left_ids.append(0)
                for token in tokens_text_left:
                    if token == '[UNK]':
                        token = 'unk'
                    tokens_left.append(token)
                    segment_left_ids.append(0)
                tokens_left.append("</s>")
                segment_left_ids.append(0)
                input_left_ids = tokenizer.convert_tokens_to_ids(tokens_left)
                input_left_mask = [1] * len(input_left_ids)
                while len(input_left_ids) < max_seq_length:
                    input_left_ids.append(0)
                    input_left_mask.append(0)
                    segment_left_ids.append(0)
                assert len(input_left_ids) == max_seq_length
                assert len(input_left_mask) == max_seq_length
                assert len(segment_left_ids) == max_seq_length


            if tokens_b:
                for token in tokens_b:
                    if token == '[UNK]':
                        token = 'unk'
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("</s>")
                segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]
            # if ex_index < 5:
            #     logger.info("*** Example ***")
            #     logger.info("guid: %s" % (example.guid))
            #     logger.info("tokens: %s" % " ".join(
            #         [tokenization.printable_text(x) for x in tokens]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logger.info(
            #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #     logger.info("label: %s (id = %d)" % (example.label, label_id))

            if tokens_aspect == None:
                features.append(
                    InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id, ))
            else:
                features.append(
                    InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        input_t_ids=input_t_ids,
                        input_t_mask=input_t_mask,
                        segment_t_ids=segment_t_ids,
                        input_without_t_ids=input_without_t_ids,
                        input_without_t_mask=input_without_t_mask,
                        segment_without_t_ids=segment_without_t_ids,
                        input_left_t_ids=input_left_t_ids,
                        input_left_t_mask=input_left_t_mask,
                        segment_left_t_ids=segment_left_t_ids,
                        input_right_t_ids=input_right_t_ids,
                        input_right_t_mask=input_right_t_mask,
                        segment_right_t_ids=segment_right_t_ids,
                        input_left_ids=input_left_ids,
                        input_left_mask=input_left_mask,
                        segment_left_ids=segment_left_ids,
#                         dependency_graph_1 = dependency_graph_1,
#                         dependency_graph_2 = dependency_graph_2,
#                         dependency_graph_3 = dependency_graph_3,
#                         dependency_graph_4 = dependency_graph_4,
                ))
        return features

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        file_in = open(input_file, "rb")
        lines = []
        for line in file_in:
            lines.append(line.decode("utf-8").split("\t"))
        return lines



class RestaurantProcessor(DataProcessor):
    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "restaurant_train.raw")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "restaurant_test.raw")), "dev")

    def get_labels(self):
        """See base class."""
        if len(self.labels) == 3:
            return ['-1', '0', '1']
        elif len(self.labels) == 4:
            return ['positive', 'neutral', 'negative', 'conflict']
        else:
            return list(self.labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        j = 0
        for i in range(0, len(lines), 3):
            guid = "%s-%s" % (set_type, j)
            j += 1
            text_left, _, text_right = [s.lower().strip() for s in lines[i][0].partition("$T$")]
            aspect = lines[i + 1][0].lower().strip()
            text_a = text_left + " " + aspect + " " + text_right  # sentence
            text_b = "What do you think of the " + aspect + " of it ?"
            # text_b = aspect
            label = lines[i + 2][0].strip()  # label
            self.labels.add(label)
            text_without_aspect = text_left + " " + text_right
            text_left_with_aspect = text_left + " " + aspect
            text_right_with_aspect = aspect + " " + text_right  # Note that there is no reverse order

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, aspect=aspect,
                             text_without_target=text_without_aspect,
                             text_left_with_target=text_left_with_aspect,
                             text_right_with_target=text_right_with_aspect,
                             text_left=text_left))
        return examples


class LaptopProcessor(DataProcessor):
    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "laptop_train.raw")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "laptop_test.raw")), "dev")

    def get_labels(self):
        """See base class."""
        if len(self.labels) == 3:
            return ['-1', '0', '1']
        elif len(self.labels) == 4:
            return ['positive', 'neutral', 'negative', 'conflict']
        else:
            return list(self.labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        j = 0
        # if set_type == 'train':
        #     del_list = np.random.choice(range(0, len(lines), 3), 600, replace=False)  # Not repeated
        for i in range(0, len(lines), 3):
            # if set_type == 'train' and i in del_list:
            #     continue
            guid = "%s-%s" % (set_type, j)
            j += 1
            text_left, _, text_right = [s.lower().strip() for s in lines[i][0].partition("$T$")]
            aspect = lines[i + 1][0].lower().strip()
            text_a = text_left + " " + aspect + " " + text_right  # sentence
            text_b = "What do you think of the " + aspect + " of it ?"
            label = lines[i + 2][0].strip()  # label
            self.labels.add(label)
            text_without_aspect = text_left + " " + text_right
            text_left_with_aspect = text_left + " " + aspect
            text_right_with_aspect = aspect + " " + text_right  # Note that there is no reverse order

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, aspect=aspect,
                             text_without_target=text_without_aspect,
                             text_left_with_target=text_left_with_aspect,
                             text_right_with_target=text_right_with_aspect,
                             text_left=text_left))
        return examples

class MamsProcessor(DataProcessor):
    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.raw")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "validation.raw")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.raw")), "dev")

    def get_labels(self):
        """See base class."""
        if len(self.labels) == 3:
            return ['-1', '0', '1']
        elif len(self.labels) == 4:
            return ['positive', 'neutral', 'negative', 'conflict']
        else:
            return list(self.labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        j = 0
        # if set_type == 'train':
        #     del_list = np.random.choice(range(0, len(lines), 3), 600, replace=False)  # Not repeated
        for i in range(0, len(lines), 3):
            # if set_type == 'train' and i in del_list:
            #     continue
            guid = "%s-%s" % (set_type, j)
            j += 1
            text_left, _, text_right = [s.lower().strip() for s in lines[i][0].partition("$T$")]
            aspect = lines[i + 1][0].lower().strip()
            text_a = text_left + " " + aspect + " " + text_right  # sentence
            text_b = "What do you think of the " + aspect + " of it ?"
            label = lines[i + 2][0].strip()  # label
            self.labels.add(label)
            text_without_aspect = text_left + " " + text_right
            text_left_with_aspect = text_left + " " + aspect
            text_right_with_aspect = aspect + " " + text_right  # Note that there is no reverse order

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, aspect=aspect,
                             text_without_target=text_without_aspect,
                             text_left_with_target=text_left_with_aspect,
                             text_right_with_target=text_right_with_aspect,
                             text_left=text_left))
        return examples
    
class TweetProcessor(DataProcessor):
    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.raw")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.raw")), "dev")

    def get_labels(self):
        """See base class."""
        if len(self.labels) == 3:
            return ['-1', '0', '1']
        else:
            return list(self.labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        j = 0
        for i in range(0, len(lines), 3):
            guid = "%s-%s" % (set_type, j)
            j += 1
            text_left, _, text_right = [s.lower().strip() for s in lines[i][0].partition("$T$")]
            aspect = lines[i + 1][0].lower().strip()
            text_a = text_left + " " + aspect + " " + text_right  # Sentence
            text_b = "What do you think of the " + aspect + " of it ?"
            label = lines[i + 2][0].strip()  # Label
            self.labels.add(label)
            text_without_aspect = text_left + " " + text_right
            text_left_with_aspect = text_left + " " + aspect
            text_right_with_aspect = aspect + " " + text_right  # Note that there is no reverse order

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, aspect=aspect,
                             text_without_target=text_without_aspect,
                             text_left_with_target=text_left_with_aspect,
                             text_right_with_target=text_right_with_aspect,
                             text_left=text_left))
        return examples

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 input_t_ids, input_t_mask, segment_t_ids,
                 input_without_t_ids, input_without_t_mask, segment_without_t_ids,
                 input_left_t_ids, input_left_t_mask, segment_left_t_ids,
                 input_right_t_ids, input_right_t_mask, segment_right_t_ids,
                 input_left_ids, input_left_mask, segment_left_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_t_ids = input_t_ids
        self.input_t_mask = input_t_mask
        self.segment_t_ids = segment_t_ids
        self.input_without_t_ids = input_without_t_ids
        self.input_without_t_mask = input_without_t_mask
        self.segment_without_t_ids = segment_without_t_ids
        self.input_left_t_ids = input_left_t_ids
        self.input_left_t_mask = input_left_t_mask
        self.segment_left_t_ids = segment_left_t_ids
        self.input_right_t_ids = input_right_t_ids
        self.input_right_t_mask = input_right_t_mask
        self.segment_right_t_ids = segment_right_t_ids
        self.input_left_ids = input_left_ids
        self.input_left_mask = input_left_mask
        self.segment_left_ids = segment_left_ids


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, aspect=None, text_without_target=None,
                 text_left_with_target=None, text_right_with_target=None, text_left=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.aspect = aspect  # add by gzj
        self.text_without_target = text_without_target  # add by gzj
        self.text_left_with_target = text_left_with_target  # add by gzj
        self.text_right_with_target = text_right_with_target  # add by gzj
        self.text_left = text_left
