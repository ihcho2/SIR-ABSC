import tokenization_word_roberta as tokenization_roberta
import tokenization_word_bert as tokenization_bert
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from data_utils_roberta import *
from bucket_iterator import BucketIterator
import pickle
from transformers import BertTokenizer, RobertaTokenizer
import os
import spacy
import time

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
        if opt.task_name == 'restaurant':
            opt.data_dir = '/home/ikhyuncho23/GoBERTa/datasets/semeval14/restaurants'
        elif opt.task_name == 'laptop':
            opt.data_dir = '/home/ikhyuncho23/GoBERTa/datasets/semeval14/laptops'
        elif opt.task_name == 'tweet':
            opt.data_dir = '/home/ikhyuncho23/GoBERTa/datasets/acl-14-short-data'
        elif opt.task_name == 'mams':
            opt.data_dir = '/home/ikhyuncho23/GoBERTa/datasets/MAMS-ATSA'
            
            
        if os.path.exists(opt.data_dir + '/train.raw'):
            train_raw = open(opt.data_dir +'/train.raw', 'r', encoding='utf-8', newline='\n', errors='ignore')
        elif os.path.exists(opt.data_dir + '/laptop_train.raw'):
            train_raw = open(opt.data_dir +'/laptop_train.raw', 'r', encoding='utf-8', newline='\n', errors='ignore')
        elif os.path.exists(opt.data_dir + '/restaurant_train.raw'):
            train_raw = open(opt.data_dir +'/restaurant_train.raw', 'r', encoding='utf-8', newline='\n', errors='ignore') 
            
        self.train_raw = train_raw.readlines()
        
        self.opt = opt
        self.train_examples = opt.processor.get_train_examples(opt.data_dir)
        self.eval_examples = opt.processor.get_dev_examples(opt.data_dir)
        if opt.task_name == 'mams':
            self.validation_examples = opt.processor.get_validation_examples(opt.data_dir)
        self.label_list = opt.processor.get_labels()
        
        if 'bert_' in opt.model_name:
            self.tokenizer = tokenization_bert.FullTokenizer(vocab_file=opt.vocab_file, do_lower_case=opt.do_lower_case)
        elif 'roberta' in opt.model_name:
            self.tokenizer = tokenization_roberta.FullTokenizer(vocab_file=opt.vocab_file, do_lower_case=opt.do_lower_case)
        ######################
        print('-'*100)
        print("Combining with dataloader from DGEDT (e.g. elements like dependency graphs).")
        print('-'*100)
        if opt.task_name == 'laptop':
            dgedt_dataset = 'lap14'
        elif opt.task_name == 'restaurant':
            dgedt_dataset = 'rest14'
        elif opt.task_name == 'tweet':
            dgedt_dataset = 'twitter'
        elif opt.task_name == 'mams':
            dgedt_dataset = 'mams'
            
        if 'bert_' in self.opt.model_name:
            absa_dataset=pickle.load(open(dgedt_dataset+'_datas_bert_'+ self.opt.parser_info+ '.pkl', 'rb'))
            self.absa_dataset = pickle.load(open(dgedt_dataset+'_datas_bert_'+ self.opt.parser_info+ '.pkl', 'rb'))
        elif 'roberta' in self.opt.model_name:
            absa_dataset=pickle.load(open(dgedt_dataset+'_datas_roberta_'+ self.opt.parser_info+ '.pkl', 'rb'))
            self.absa_dataset = pickle.load(open(dgedt_dataset+'_datas_roberta_'+ self.opt.parser_info+ '.pkl', 'rb'))
            
        if 'spacy_sm' in self.opt.parser_info:
            self.nlp = spacy.load('en_core_web_sm')
        elif 'spacy_lg' in self.opt.parser_info:
            self.nlp = spacy.load('en_core_web_lg')
            
        with open('./datasets/' + self.opt.parser_info + '.global_edge_vocab', 'rb') as file:
            self.edge_vocab_ = pickle.load(file)
        with open('./datasets/' + self.opt.parser_info + '.global_pos_vocab', 'rb') as file:
            self.pos_vocab_ = pickle.load(file)
            
        self.edge_vocab = {}
        self.pos_vocab = {}
        
        for key in self.edge_vocab_.keys():
            self.edge_vocab[self.edge_vocab_[key]] = key
        for key in self.pos_vocab_.keys():
            self.pos_vocab[self.pos_vocab_[key]] = key
        
        print('-'*77)
        print(f"* Loaded global edge vocab from {'./datasets/' + self.opt.parser_info + '.global_edge_vocab'}")
        print(f"* Loaded global pos vocab from {'./datasets/' + self.opt.parser_info + '.global_pos_vocab'}")
           
        print('-'*77)
        print('POS_vocab: ', self.pos_vocab)
        print('-'*77)
        
        if os.path.exists(f'./datasets/{self.opt.parser_info}_global_DEP_info_emb'):
            with open(f'./datasets/{self.opt.parser_info}_global_DEP_info_emb', 'rb') as file:
                self.dep_vocab = pickle.load(file)
        else:
            self.dep_vocab = {'None': 0, '[s] or [CLS]': 1, '[g]': 2}
            self.create_new_global_dep()
                
        self.idx2dep = {}
        for key in self.dep_vocab.keys():
            self.idx2dep[self.dep_vocab[key]] = key     
            
        print('self.idx2dep: ', self.idx2dep)
        print('-'*77)
        
        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=100000, max_seq_length = self.opt.max_seq_length, shuffle=True, input_format = self.opt.input_format, model_name = self.opt.model_name)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=100000, max_seq_length = self.opt.max_seq_length, shuffle=False, input_format = self.opt.input_format, model_name = self.opt.model_name)
        if opt.task_name == 'mams':
            self.eval_data_loader = BucketIterator(data=absa_dataset.validation_data, batch_size=100000, max_seq_length = self.opt.max_seq_length, shuffle=False, input_format = self.opt.input_format, model_name = self.opt.model_name)
        
        self.DGEDT_train_data = self.train_data_loader.data
        self.DGEDT_train_batches = self.train_data_loader.batches
        
        self.DGEDT_test_data = self.test_data_loader.data
        self.DGEDT_test_batches = self.test_data_loader.batches
        
#         eval_text = []
#         eval_aspect = []
#         for i in range(len(self.DGEDT_test_data)):
#             eval_text.append(self.DGEDT_test_data[i]['text'])
#             eval_aspect.append(self.DGEDT_test_data[i]['aspect'])
        
#         with open(f'./analysis/rest_eval_text.pkl', 'wb') as file:
#             pickle.dump(eval_text, file)
            
#         with open('./analysis/rest_eval_aspect.pkl', 'wb') as file:
#             pickle.dump(eval_aspect, file)
        
        if opt.task_name == 'mams':
            self.DGEDT_validation_data = self.eval_data_loader.data
            self.DGEDT_validation_batches = self.eval_data_loader.batches
        
            
        if os.path.exists(f'./Cache/{self.opt.task_name}_train_gcls_attention_mask.pt'):
            time0 = time.time()
            self.train_gcls_attention_mask = torch.load(f'./Cache/{self.opt.task_name}_train_gcls_attention_mask.pt')
            self.train_VDC_info = torch.load(f'./Cache/{self.opt.task_name}_train_VDC_info.pt')
            self.train_DEP_info = torch.load(f'./Cache/{self.opt.task_name}_train_DEP_info.pt')
            self.train_POS_info = torch.load(f'./Cache/{self.opt.task_name}_train_POS_info.pt')
            self.train_hVDC = torch.load(f'./Cache/{self.opt.task_name}_train_hVDC.pt')
            
            self.eval_gcls_attention_mask = torch.load(f'./Cache/{self.opt.task_name}_eval_gcls_attention_mask.pt')
            self.eval_VDC_info = torch.load(f'./Cache/{self.opt.task_name}_eval_VDC_info.pt')
            self.eval_DEP_info = torch.load(f'./Cache/{self.opt.task_name}_eval_DEP_info.pt')
            self.eval_POS_info = torch.load(f'./Cache/{self.opt.task_name}_eval_POS_info.pt')
            self.eval_hVDC = torch.load(f'./Cache/{self.opt.task_name}_eval_hVDC.pt')
            
            if self.opt.task_name == 'mams':
                self.validation_gcls_attention_mask = torch.load(f'./Cache/{self.opt.task_name}_validation_gcls_attention_mask.pt')
                self.validation_VDC_info = torch.load(f'./Cache/{self.opt.task_name}_validation_VDC_info.pt')
                self.validation_DEP_info = torch.load(f'./Cache/{self.opt.task_name}_validation_DEP_info.pt')
                self.validation_POS_info = torch.load(f'./Cache/{self.opt.task_name}_validation_POS_info.pt')
                self.validation_hVDC = torch.load(f'./Cache/{self.opt.task_name}_validation_hVDC.pt')
                
            print('Time took to load the required tensors: ', time.time()-time0)
            
        else:
            self.train_gcls_attention_mask, self.train_VDC_info, self.train_DEP_info, self.train_POS_info, self.train_hVDC = \
            self.process_DG(self.DGEDT_train_data, data_type = 'train_data')
        
            self.eval_gcls_attention_mask, self.eval_VDC_info, self.eval_DEP_info, self.eval_POS_info, self.eval_hVDC = \
            self.process_DG(self.DGEDT_test_data, data_type = 'eval_data')
        
            if opt.task_name == 'mams':
                self.validation_gcls_attention_mask, self.validation_VDC_info, self.validation_DEP_info, self.validation_POS_info, self.validation_hVDC \
                = self.process_DG(self.DGEDT_validation_data, data_type = 'validation_data')
                
        ######################
        self.train_data, self.train_dataloader, self.train_tran_indices, self.train_span_indices, \
        self.train_extended_attention_mask, self.train_current_VDC  = self.get_data_loader(examples=self.train_examples, 
                                                                                           type='train_data', 
                                                                   gcls_attention_mask =self.train_gcls_attention_mask,
                                                                   hVDC = self.train_hVDC)
        
        self.eval_data, self.eval_dataloader, self.eval_tran_indices, self.eval_span_indices, \
        self.eval_extended_attention_mask, self.eval_current_VDC = self.get_data_loader(examples=self.eval_examples, type='eval_data', 
                                                                 gcls_attention_mask = self.eval_gcls_attention_mask, 
                                                                 hVDC = self.eval_hVDC)
        
        if opt.task_name == 'mams':
            self.validation_data, self.validation_dataloader, self.validation_tran_indices, self.validation_span_indices, \
            self.validation_extended_attention_mask, self.validation_current_VDC= self.get_data_loader(examples=self.validation_examples, type='validation', gcls_attention_mask = self.validation_gcls_attention_mask, hVDC = self.validation_hVDC)
            
    def create_new_global_dep(self):
        for dataset in ['lap14', 'rest14', 'twitter', 'mams']:
            if 'bert_' in self.opt.model_name:
                dataset_ = pickle.load(open(dataset+'_datas_bert_' + self.opt.parser_info +'.pkl', 'rb'))
            elif 'roberta' in self.opt.model_name:
                dataset_ = pickle.load(open(dataset+'_datas_roberta_' + self.opt.parser_info +'.pkl', 'rb'))
                
            train_data_loader = BucketIterator(data=dataset_.train_data, batch_size=100000, max_seq_length = self.opt.max_seq_length, shuffle=True, input_format = self.opt.input_format, model_name = self.opt.model_name)
            
            self.get_dep_labels(train_data_loader.data)
            
            print('len(self.dep_vocab): ', len(self.dep_vocab))
            
        with open(f'./datasets/{self.opt.parser_info}_global_DEP_info_emb', 'wb') as file:
            pickle.dump(self.dep_vocab, file)
        
        
    def get_dep_labels(self, DGEDT_train_data):
        edge_merge = {'self': 'self', '<pad>': '<pad>', 'punct': 'punct', 'nsubj': 'sbj', 'det': 'dep', 'prep': 'prep', 
                      'pobj': 'arg', 'advmod': 'mod', 'amod': 'mod', 'conj': 'conj', 'dobj': 'dobj', 'aux': 'aux', 'cc': 'cc',
                      'compound': 'mod', 'acomp': 'arg', 'advcl': 'mod', 'ccomp': 'arg', 'mark': 'dep', 'xcomp': 'arg', 
                      'poss': 'mod', 'nummod': 'mod', 'relcl': 'mod', 'neg': 'neg', 'attr': 'arg', 'prt': 'dep', 
                      'npadvmod': 'mod', 'auxpass': 'aux', 'pcomp': 'arg', 'nmod': 'mod', 'nsubjpass': 'sbj', 'acl': 'mod', 
                      'appos': 'mod', 'quantmod': 'mod', 'dep': 'dep', 'expl': 'sbj', 'predet': 'mod', 'dative': 'arg',
                      'intj': 'dep', 'case': 'dep', 'oprd': 'arg', 'csubj': 'sbj', 'parataxis': 'dep', 'agent': 'arg',
                      'preconj': 'mod', 'meta': 'dep'}
        
        AOBG_label = []
        
        for i in range(len(DGEDT_train_data)):
            AOBG_label.append({})
            dg = DGEDT_train_data[i]['dependency_graph'][0] + DGEDT_train_data[i]['dependency_graph'][1]
            dg[dg>=1] = 1
            dg = torch.tensor(dg)
            
            edge_1 = DGEDT_train_data[i]['dependency_graph'][2] 
            edge_2 = DGEDT_train_data[i]['dependency_graph'][3] 
            
            for j in range(dg.size(0)): 
                AOBG_label[i][j] = {}
                AOBG_label[i][j]['dep_label'] = 'None' 
                
            used_trans = []
            last_trans = []
            for k in range(len(DGEDT_train_data[i]['span_indices'])): 
                tran_start = DGEDT_train_data[i]['span_indices'][k][0]  # The starting tran_id (=word) of the target.
                tran_end = DGEDT_train_data[i]['span_indices'][k][1]    # The ending tran_id (=word) of the target.
                
                for item in range(tran_start, tran_end):
                    AOBG_label[i][item]['dep_label'] = 'self'
                    used_trans.append(item)
                    last_trans.append(item)
                    
            edge_merge = {'self': 'self', '<pad>': '<pad>', 'punct': 'punct', 'nsubj': 'sbj', 'det': 'dep', 'prep': 'prep', 
                      'pobj': 'arg', 'advmod': 'mod', 'amod': 'mod', 'conj': 'conj', 'dobj': 'dobj', 'aux': 'aux', 'cc': 'cc',
                      'compound': 'mod', 'acomp': 'arg', 'advcl': 'mod', 'ccomp': 'arg', 'mark': 'dep', 'xcomp': 'arg', 
                      'poss': 'mod', 'nummod': 'mod', 'relcl': 'mod', 'neg': 'neg', 'attr': 'arg', 'prt': 'dep', 
                      'npadvmod': 'mod', 'auxpass': 'aux', 'pcomp': 'arg', 'nmod': 'mod', 'nsubjpass': 'sbj', 'acl': 'mod', 
                      'appos': 'mod', 'quantmod': 'mod', 'dep': 'dep', 'expl': 'sbj', 'predet': 'mod', 'dative': 'arg',
                      'intj': 'dep', 'case': 'dep', 'oprd': 'arg', 'csubj': 'sbj', 'parataxis': 'dep', 'agent': 'arg',
                      'preconj': 'mod', 'meta': 'dep'}
            
            for l in range(1, 12):
                last_trans_ = []
                for item in last_trans:
                    x = (dg[item] == 1).nonzero(as_tuple=True)[0]
                    for item_ in x:
                        if int(item_) not in used_trans:
                            used_trans.append(int(item_))
                            last_trans_.append(int(item_))
                            if edge_1[item][int(item_)] != 0 and edge_2[item][int(item_)] != 0:
                                print('Something wrong')
                            elif edge_1[item][int(item_)] != 0:
                                if l < 3:
#                                     AOBG_label[i][int(item_)]['dep_label'] = AOBG_label[i][int(item)]['dep_label'] +';+'+\
#                                                                              edge_merge[self.edge_vocab[edge_1[item]
#                                                                                                         [int(item_)]]]
                                    AOBG_label[i][int(item_)]['dep_label'] = AOBG_label[i][int(item)]['dep_label'] +';+'+\
                                                                             self.edge_vocab[edge_1[item]
                                                                                                        [int(item_)]]
                                else:
                                    AOBG_label[i][int(item_)]['dep_label'] = f'{l}_con'
                            elif edge_2[item][int(item_)] != 0:
                                if l < 3:
#                                     AOBG_label[i][int(item_)]['dep_label'] = AOBG_label[i][int(item)]['dep_label'] +';-'+\
#                                                                              edge_merge[self.edge_vocab[edge_2[item]
#                                                                                                         [int(item_)]]]
                                    AOBG_label[i][int(item_)]['dep_label'] = AOBG_label[i][int(item)]['dep_label'] +';-'+\
                                                                             self.edge_vocab[edge_2[item]
                                                                                                        [int(item_)]]
                                else:
                                    AOBG_label[i][int(item_)]['dep_label'] = f'{l}_con'
                            else:
                                print('Something wrong')
                                
                last_trans = last_trans_
                
        for i in range(len(DGEDT_train_data)):
            for j in range(len(AOBG_label[i])):
                if AOBG_label[i][j]['dep_label'] not in self.dep_vocab.keys():
                    self.dep_vocab[AOBG_label[i][j]['dep_label']] = len(self.dep_vocab)
        
                  
    def process_DG(self, DGEDT_train_data, data_type = 'train_data'):
        total_tokens = [] # Used for doing DG analysis
        total_words = []
        
        AOBG_label = []
        hVDC_word = []
        hVDC_token = []
        second_g_idx = []
        
        VDC_info = torch.full([len(DGEDT_train_data), self.opt.max_seq_length], 99) 
        DEP_info = torch.full([len(DGEDT_train_data), self.opt.max_seq_length], 0) 
        POS_info = torch.full([len(DGEDT_train_data), self.opt.max_seq_length], 99) 
        
        surface_VDC_info_token_level = torch.full([len(DGEDT_train_data), self.opt.max_seq_length], 99.0) 
        surface_VDC_info_word_level = torch.full([len(DGEDT_train_data), self.opt.max_seq_length], 99.0) # 같은 word에서 나온 토큰은 같은 distance 적용.
        
        gcls_attention_mask = torch.zeros((len(DGEDT_train_data), 12, 128), dtype=torch.float)
        gcls_attention_mask_surface_distance_word_level = torch.zeros((len(DGEDT_train_data), 12, 128), dtype=torch.float)
        gcls_attention_mask_surface_distance_token_level = torch.zeros((len(DGEDT_train_data), 12, 128), dtype=torch.float)
        
        # Just in case we might need to use VDC_info in word level (e.g., hVDC).
        number_of_words_L = torch.zeros((len(DGEDT_train_data), 12), dtype=torch.float) # distance L에 몇 개의 word
#         number_of_tokens_L = torch.zeros((len(DGEDT_train_data), 12), dtype=torch.float) # distance L에 몇 개의 token
        
        for i in range(len(DGEDT_train_data)):
            AOBG_label.append({})
            
            dg = DGEDT_train_data[i]['dependency_graph'][0] + DGEDT_train_data[i]['dependency_graph'][1]
            dg[dg>=1] = 1
            dg_in = DGEDT_train_data[i]['dependency_graph'][0]
            dg_out = DGEDT_train_data[i]['dependency_graph'][1]
            dg = torch.tensor(dg)
            
            edge_1 = DGEDT_train_data[i]['dependency_graph'][2] 
            edge_2 = DGEDT_train_data[i]['dependency_graph'][3] 
            
            sep_pos = len(DGEDT_train_data[i]['text_indices']) 
            
            total_tokens.append(sep_pos-2)
            second_g_idx.append(sep_pos+2)
            total_words.append(dg.size(0))

#             if len(DGEDT_train_data[i]['span_indices']) != 1:
#                 print('Multiple same aspects in the sentence : ', i)
#             assert len(DGEDT_train_data[i]['span_indices']) == 1 # For Lap14, Rest14, and MAMS.
    
            target_begin_word_idx = DGEDT_train_data[i]['span_indices'][0][0] # Beginning of the target aspect in word-level.
            target_end_word_idx = DGEDT_train_data[i]['span_indices'][0][1] # End of the target aspect in word-level.
            
            ##########
            if self.opt.input_format == 'td_X':
                A = 1
            elif 'g_infront_of' in self.opt.input_format:
                A = 2
            elif 'g' in self.opt.input_format:
                A = 2
            elif self.opt.input_format == 'X':
                A = 2
                
            target_begin_token_idx = DGEDT_train_data[i]['tran_indices'][target_begin_word_idx][0] + A
            target_end_token_idx = DGEDT_train_data[i]['tran_indices'][target_end_word_idx-1][1] + A
            # might need to change this when using Twitter since there could be multiple aspects.
            
            ## Initialize AOBG labels
            
            # 나중에 지울 것들 
            assert dg.size(0) == len(DGEDT_train_data[i]['tran_indices'])
            
            # Iterating over all words (parsed by the parser's tokenizer).
            for j in range(dg.size(0)): 
                AOBG_label[i][j] = {}
                AOBG_label[i][j]['token_idx_start'] = 0 
                AOBG_label[i][j]['token_idx_end'] = 0
                
                if j < target_begin_word_idx:
                    AOBG_label[i][j]['surface_distance'] = target_begin_word_idx - j # word-level distance.
                elif target_begin_word_idx <= j and j <target_end_word_idx:
                    AOBG_label[i][j]['surface_distance'] = 0
                else:
                    AOBG_label[i][j]['surface_distance'] = j - target_end_word_idx + 1
                    
                AOBG_label[i][j]['syn_distance'] = 99
                AOBG_label[i][j]['dep_label'] = 'None' 
                AOBG_label[i][j]['pos_tag'] = DGEDT_train_data[i]['dependency_graph'][-2][j]
                AOBG_label[i][j]['is_root'] = DGEDT_train_data[i]['dependency_graph'][-1][j]
                
                # Define A
                if self.opt.input_format == 'TD_X':
                    A = 1
                elif 'g_infront_of' in self.opt.input_format:
                    if j >= target_begin_word_idx:
                        A = 2
                    else:
                        A = 1
                elif 'g' in self.opt.input_format:
                    A = 2
                elif self.opt.input_format == 'X':
                    A = 2
                
                AOBG_label[i][j]['token_idx_start'] = DGEDT_train_data[i]['tran_indices'][j][0]+A
                AOBG_label[i][j]['token_idx_end'] = DGEDT_train_data[i]['tran_indices'][j][1]+A
                
            # Now filling in the 'syn_distance', 'dep_label'
            used_trans = []
            last_trans = []
#             for k in range(len(DGEDT_train_data[i]['span_indices'])): 
            for k in range(1): 
                tran_start = DGEDT_train_data[i]['span_indices'][k][0]  # The starting tran_id (=word) of the target.
                tran_end = DGEDT_train_data[i]['span_indices'][k][1]    # The ending tran_id (=word) of the target.
                
                for item in range(tran_start, tran_end):
                    number_of_words_L[i][0] += 1
                    
                    # 나중에 지울 것
                    assert AOBG_label[i][item]['surface_distance'] == 0
                    
                    AOBG_label[i][item]['syn_distance'] = 0  # target itself => syn_distance = 0
                    AOBG_label[i][item]['dep_label'] = 'self' 
                    used_trans.append(item)
                    last_trans.append(item)

#                     number_of_tokens_L 여기 나중에 고치기

            hVDC_found = False # word-level hVDC
            hVDC_found_token = False # token-level hVDC
            
            edge_merge = {'self': 'self', '<pad>': '<pad>', 'punct': 'punct', 'nsubj': 'sbj', 'det': 'dep', 'prep': 'prep', 
                      'pobj': 'arg', 'advmod': 'mod', 'amod': 'mod', 'conj': 'conj', 'dobj': 'dobj', 'aux': 'aux', 'cc': 'cc',
                      'compound': 'mod', 'acomp': 'arg', 'advcl': 'mod', 'ccomp': 'arg', 'mark': 'dep', 'xcomp': 'arg', 
                      'poss': 'mod', 'nummod': 'mod', 'relcl': 'mod', 'neg': 'neg', 'attr': 'arg', 'prt': 'dep', 
                      'npadvmod': 'mod', 'auxpass': 'aux', 'pcomp': 'arg', 'nmod': 'mod', 'nsubjpass': 'sbj', 'acl': 'mod', 
                      'appos': 'mod', 'quantmod': 'mod', 'dep': 'dep', 'expl': 'sbj', 'predet': 'mod', 'dative': 'arg',
                      'intj': 'dep', 'case': 'dep', 'oprd': 'arg', 'csubj': 'sbj', 'parataxis': 'dep', 'agent': 'arg',
                      'preconj': 'mod', 'meta': 'dep'}
            
            if (number_of_words_L[i][0]/total_words[i] >= self.opt.VDC_threshold) and hVDC_found == False:
                hVDC_found = True
                hVDC_word.append(0)
                    
            for l in range(1, 12):
                last_trans_ = [] 
                number_of_words_L[i][l] = number_of_words_L[i][l-1] # 누적으로 하는게 편할 듯.
#                 number_of_tokens_L[i][l] = number_of_tokens_L[i][l-1]
                for item in last_trans:
                    x = (dg[item] == 1).nonzero(as_tuple=True)[0]
                    for item_ in x:
                        if int(item_) not in used_trans:
                            
#                             # POS-guided?
#                             if AOBG_label[i][int(item_)]['pos_tag'] in [3,8]:
#                                 continue
                                
                            used_trans.append(int(item_))
                            last_trans_.append(int(item_))
                            number_of_words_L[i][l] += 1
                            
                            # 나중에 지울 것.
                            assert AOBG_label[i][int(item_)]['syn_distance'] == 99 # 중복 없음 확인.
                            AOBG_label[i][int(item_)]['syn_distance'] = l
    
                            if edge_1[item][int(item_)] != 0 and edge_2[item][int(item_)] != 0:
                                print('Something wrong')
                
                            elif edge_1[item][int(item_)] != 0:
                                if l < 3:
#                                     AOBG_label[i][int(item_)]['dep_label'] = AOBG_label[i][int(item)]['dep_label'] +';+'+\
#                                                                              edge_merge[self.edge_vocab[edge_1[item]
#                                                                                                         [int(item_)]]]
                                    AOBG_label[i][int(item_)]['dep_label'] = AOBG_label[i][int(item)]['dep_label'] +';+'+\
                                                                             self.edge_vocab[edge_1[item]
                                                                                                        [int(item_)]]
                                else:
                                    AOBG_label[i][int(item_)]['dep_label'] = f'{l}_con'
                                    
                            elif edge_2[item][int(item_)] != 0:
                                if l < 3:
#                                     AOBG_label[i][int(item_)]['dep_label'] = AOBG_label[i][int(item)]['dep_label'] +';-'+\
#                                                                              edge_merge[self.edge_vocab[edge_2[item]
#                                                                                                         [int(item_)]]]
                                    AOBG_label[i][int(item_)]['dep_label'] = AOBG_label[i][int(item)]['dep_label'] +';-'+\
                                                                             self.edge_vocab[edge_2[item]
                                                                                                        [int(item_)]]
                                else:
                                    AOBG_label[i][int(item_)]['dep_label'] = f'{l}_con'
                            else:
                                print('Something wrong')
                                
                last_trans = last_trans_
                    
                if (number_of_words_L[i][l]/total_words[i] > self.opt.VDC_threshold) and hVDC_found == False:
                    hVDC_found = True
                    hVDC_word.append(l)
                    
            if hVDC_found == False:
                hVDC_word.append(11)
                
            if hVDC_found_token == False:
                hVDC_token.append(11)
                
#             print('total_words[i]: ', total_words[i])
#             print('number_of_words_L[i]: ', number_of_words_L[i])
#             print('hVDC_word[i]: ', hVDC_word[i])
#             print('-'*77)
                            
            for word in range(len(AOBG_label[i])):
                start = AOBG_label[i][word]['token_idx_start']
                end = AOBG_label[i][word]['token_idx_end']
                
                # 1. Create VDC_info, surface_VDC_info_word_level, surface_VDC_info_token_level
                VDC_info[i][start:end] = AOBG_label[i][word]['syn_distance']
                surface_VDC_info_word_level[i][start:end] = AOBG_label[i][word]['surface_distance']
                
                for p in range(start, end):
                    if p < target_begin_token_idx:
                        surface_VDC_info_token_level[i][p] = target_begin_token_idx - p
                    elif p < target_end_token_idx:
                        surface_VDC_info_token_level[i][p] = 0
                    else:
                        surface_VDC_info_token_level[i][p] = p - target_end_token_idx + 1
                        
                # 2. POS_info
                POS_info[i][start:end] = AOBG_label[i][word]['pos_tag']
                
                # 3. DEP_info
                if AOBG_label[i][word]['dep_label'] not in self.dep_vocab:
                    # 나중에 지우기.
#                     assert data_type == 'train_data'
                    DEP_info[i][start:end] = 0
                else:
                    DEP_info[i][start:end] = self.dep_vocab[AOBG_label[i][word]['dep_label']]
                
                # s, g 위치 처리.
                if self.opt.input_format == 'TD_X':
                    VDC_info[i][0] = 129
                    POS_info[i][0] = len(self.pos_vocab) 
                    DEP_info[i][0] = self.dep_vocab['[s] or [CLS]']
                    
                elif 'g_infront_of' in self.opt.input_format:
                    VDC_info[i][0] = 129
                    VDC_info[i][target_begin_token_idx-1] = 999
                    VDC_info[i][second_g_idx[i]] = 130 # or 999
                    
                    POS_info[i][0] = len(self.pos_vocab)
                    POS_info[i][target_begin_token_idx-1] = len(self.pos_vocab) + 1
                    POS_info[i][second_g_idx[i]] = len(self.pos_vocab) + 2 # or len(self.pos_vocab) + 1
                    
                    DEP_info[i][0] = DEP_info[i][0] = self.dep_vocab['[s] or [CLS]']
                    DEP_info[i][target_begin_token_idx-1] = self.dep_vocab['[g]']
                    DEP_info[i][second_g_idx[i]] = len(self.dep_vocab) + 1 # or self.dep_vocab['[g]']
                    
                elif 'g' in self.opt.input_format:
                    VDC_info[i][0] = 129
                    VDC_info[i][1] = 999
                    VDC_info[i][second_g_idx[i]] = 130 # or 999
                    
                    POS_info[i][0] = len(self.pos_vocab)
                    POS_info[i][1] = len(self.pos_vocab)+1
                    POS_info[i][second_g_idx[i]] = len(self.pos_vocab) + 2 # or len(self.pos_vocab) + 1
                    
                    DEP_info[i][0] = 1 # 1
                    DEP_info[i][1] = 1 # 2
                    DEP_info[i][second_g_idx[i]] = 0 # len(self.dep_vocab) + 1 # or self.dep_vocab['[g]']
                    
                    
                elif self.opt.input_format == 'X':
                    VDC_info[i][0] = 129
                    VDC_info[i][1] = 999
#                     VDC_info[i][second_g_idx[i]] = 130 # or 999
                    
                    POS_info[i][0] = len(self.pos_vocab)
                    POS_info[i][1] = len(self.pos_vocab)+1
                    POS_info[i][second_g_idx[i]] = len(self.pos_vocab) + 2 # or len(self.pos_vocab) + 1
                    
                    DEP_info[i][0] = 1 # len(self.dep_vocab)
                    DEP_info[i][1] = 1 # self.dep_vocab['[g]']
                    DEP_info[i][second_g_idx[i]] = 0 # len(self.dep_vocab) + 1 # or self.dep_vocab['[g]']
                    
            
            for j in range(12):
                gcls_attention_mask[i][j] = (VDC_info[i]<=j).float()
                gcls_attention_mask_surface_distance_word_level[i][j] = (surface_VDC_info_word_level[i]<=j).float()
                gcls_attention_mask_surface_distance_token_level[i][j] = (surface_VDC_info_token_level[i]<=j).float()
                
                # s,g 처리
                gcls_attention_mask[i][j][0] = 1.0
                gcls_attention_mask_surface_distance_word_level[i][j][0] = 1.0
                gcls_attention_mask_surface_distance_token_level[i][j][0] = 1.0
                
                if 'g_infront_of' in self.opt.input_format:
                    gcls_attention_mask[i][j][target_begin_token_idx-1] = 1.0
                    gcls_attention_mask_surface_distance_word_level[i][j][target_begin_token_idx-1] = 1.0
                    gcls_attention_mask_surface_distance_token_level[i][j][target_begin_token_idx-1] = 1.0
                    
                elif 'g' in self.opt.input_format:
                    gcls_attention_mask[i][j][1] = 1.0
                    gcls_attention_mask_surface_distance_word_level[i][j][1] = 1.0
                    gcls_attention_mask_surface_distance_token_level[i][j][1] = 1.0
                    
                elif self.opt.input_format == 'X':
                    gcls_attention_mask[i][j][1] = 1.0
                    gcls_attention_mask_surface_distance_word_level[i][j][1] = 1.0
                    gcls_attention_mask_surface_distance_token_level[i][j][1] = 1.0
        
        
        os.makedirs('./Cache', exist_ok=True)
        data_t = {'train_data': 'train', 'eval_data': 'eval', 'validation_data': 'validation'}
        torch.save(gcls_attention_mask, f'./Cache/{self.opt.task_name}_{data_t[data_type]}_gcls_attention_mask.pt')
        torch.save(VDC_info, f'./Cache/{self.opt.task_name}_{data_t[data_type]}_VDC_info.pt')
        torch.save(DEP_info, f'./Cache/{self.opt.task_name}_{data_t[data_type]}_DEP_info.pt')
        torch.save(POS_info, f'./Cache/{self.opt.task_name}_{data_t[data_type]}_POS_info.pt')
        torch.save(hVDC_word, f'./Cache/{self.opt.task_name}_{data_t[data_type]}_hVDC.pt')
            
            
        return gcls_attention_mask, VDC_info, DEP_info, POS_info, hVDC_word

            
    def get_data_loader(self, examples, type='train_data', gcls_attention_mask=None, hVDC=None):
        features = self.convert_examples_to_features(
            examples, self.label_list, self.opt.max_seq_length, self.tokenizer)
        
        if type == 'train_data':
            DGEDT_batches = self.DGEDT_train_batches[0]
            DGEDT_data = self.DGEDT_train_data    # 점검용
        elif type == 'eval_data':
            DGEDT_batches = self.DGEDT_test_batches[0]
            DGEDT_data = self.DGEDT_test_data    # 점검용
        elif type == 'validation':
            DGEDT_batches = self.DGEDT_validation_batches[0]
            DGEDT_data = self.DGEDT_validation_data 
        
        batch_size_ = DGEDT_batches['text_indices'].size(0)
        
        all_input_ids_org = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        
        assert all_input_ids_org.size(0) == batch_size_
        ##############################
        all_input_ids = DGEDT_batches['text_indices']
        
        
        ##############################

#         all_graph_s_pos = DGEDT_batches['graph_s_pos']

        all_input_mask_org = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        ##############################
        text_len = torch.sum(DGEDT_batches['text_indices'] != 1, dim=-1)
        
        
        all_input_mask = length2mask(text_len, DGEDT_batches['text_indices'].size(1))
        
        all_segment_ids_org = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        ##############################
        all_segment_ids = all_segment_ids_org  # all zeroes of size 128.
        
        ########## When target is appended at the end or the beginning.
        for i in range(batch_size_):
            x = (all_input_ids[i] == 2).nonzero(as_tuple=True)[0]
            all_segment_ids[i][x[0]+1:x[-1]+1] = 1
            
        ##########
        
    
        all_label_ids_org = torch.tensor([f.label_id for f in features], dtype=torch.long)
        ##############################
        all_label_ids = all_label_ids_org
        for i in range(batch_size_):
            assert all_label_ids[i] == DGEDT_batches['polarity'][i]
        # Note that DGEDT, TD-BERT guid 순서는 동일.
        # No change. Quite obviously, the labels are the same.
        ##############################
        
        all_tran_indices = DGEDT_batches['tran_indices']
        all_span_indices = DGEDT_batches['span_indices']
        all_input_guids = torch.tensor([i for i in range(batch_size_)], dtype = torch.long)
        all_input_dg = DGEDT_batches['dependency_graph']
        all_input_dg1 = DGEDT_batches['dependency_graph1']
        all_input_dg2 = DGEDT_batches['dependency_graph2']
        all_input_dg3 = DGEDT_batches['dependency_graph3']
        
        extended_attention_mask = torch.zeros([all_input_ids.size(0),1,1,128])    # torch.Size([32, 1, 1, 128])
        print('extended_attention_mask.size() should be [N,1,1,128] ', extended_attention_mask.size())
        extended_attention_mask = extended_attention_mask.repeat(1,1,all_input_ids.size(1),1)    # torch.Size([32, 1, 128, 128])
        print('extended_attention_mask.size() should be [N,1,128,128] ', extended_attention_mask.size())

        extended_attention_mask = extended_attention_mask.unsqueeze(1).repeat(1,12,1,1,1)
        print('extended_attention_mask.size() should be [N,12,1,128,128] ', extended_attention_mask.size())
        
        g_config = []
        if self.opt.g_config == None:
            g_config = [1,1,1,1]
        elif len(self.opt.g_config) == 4:
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
        
        new_VDC_k = [[]]*all_input_ids.size(0)
        current_VDC = torch.zeros([all_input_ids.size(0), 12], dtype= torch.float)
        print('='*77)
        print('VDC threshold: ', self.opt.VDC_threshold)
        if self.opt.constant_vdc != None:
            print('self.opt.constant_vdc: ', self.opt.constant_vdc)
            
        if self.opt.VDC_threshold != None and self.opt.use_hVDC == True:
            for i in range(all_input_ids.size(0)):
                if hVDC[i] == 0: 
                    new_VDC_k = [0] * 12
                elif hVDC[i] == 1:
                    new_VDC_k = [0,0,0,0,0,0,1,1,1,1,1,1]
                elif hVDC[i] == 2:
                    new_VDC_k = [0,0,0,0,1,1,1,1,2,2,2,2]
                elif hVDC[i] == 3:
                    new_VDC_k = [0,0,0,1,1,1,2,2,2,3,3,3]
                elif hVDC[i] == 4:
                    new_VDC_k = [0,0,0,1,1,1,2,2,3,3,4,4]
#                     new_VDC_k = [0,0,1,1,2,2,3,3,3,4,4,4]
                elif hVDC[i] == 5:
                    new_VDC_k = [0,0,1,1,2,2,3,3,4,4,5,5]
                elif hVDC[i] == 6:
                    new_VDC_k = [0,0,1,1,2,2,3,3,4,4,5,6]
#                     new_VDC_k = [0,1,2,2,3,3,4,4,5,5,6,6]
                elif hVDC[i] == 7:
                    new_VDC_k = [0,0,1,1,2,2,3,3,4,5,6,7]
#                     new_VDC_k = [0,1,2,3,4,4,5,5,6,6,7,7]
                elif hVDC[i] == 8:
                    new_VDC_k = [0,0,1,1,2,2,3,4,5,6,7,8]
#                     new_VDC_k = [0,1,2,3,4,5,6,6,7,7,8,8]
                elif hVDC[i] == 9:
                    new_VDC_k = [0,0,1,1,2,3,4,5,6,7,8,9]
#                     new_VDC_k = [0,1,2,3,4,5,6,7,8,8,9,9]
                elif hVDC[i] == 10:
                    new_VDC_k = [0,0,1,2,3,4,5,6,7,8,9,10]
#                     new_VDC_k = [0,1,2,3,4,5,6,7,8,9,10,10]
                elif hVDC[i] == 11:
                    new_VDC_k = [0,1,2,3,4,5,6,7,8,9,10,11]
                    
                if self.opt.do_auto == True:
                    current_VDC[i][:] = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11], dtype = torch.float)
                    for j, item in enumerate([0,1,2,3,4,5,6,7,8,9,10,11]):
                        extended_attention_mask[i, j, 0, 1, :] =  (1 - gcls_attention_mask[i][item]) * -10000.0
                else:
                    current_VDC[i][:] = torch.tensor(new_VDC_k, dtype = torch.float)
                    for j, item in enumerate(new_VDC_k):
                        extended_attention_mask[i, j, 0, 1, :] =  (1 - gcls_attention_mask[i][item]) * -10000.0
                
        elif self.opt.use_hVDC == False:
            for i in range(all_input_ids.size(0)):
                if self.opt.do_auto == True:
                    current_VDC[i][:] = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11], dtype = torch.float)
                    for j, item in enumerate([0,1,2,3,4,5,6,7,8,9,10,11]):
                        extended_attention_mask[i, j, 0, 1, :] =  (1 - gcls_attention_mask[i][item]) * -10000.0
                else:
                    current_VDC[i] = torch.tensor(self.opt.constant_vdc)
                    for j, item in enumerate(self.opt.constant_vdc):
                        extended_attention_mask[i, j, 0, 1, :] =  (1 - gcls_attention_mask[i][item]) * -10000.0
                        
        print('new_VDC_K statistics: ')
        max_k = max(hVDC)
        for jj in range(max_k+1):
            print(f'VDC={jj}: {100*float(sum(torch.tensor(hVDC) == jj)/len(hVDC))}%')  
        
        
        ###############
        if 'bert_' in self.opt.model_name:
            data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_guids)
        elif 'roberta' in self.opt.model_name:
            data = TensorDataset(all_input_ids, all_label_ids, all_input_guids)
        
        if type == 'train_data':
            train_data = data
            train_sampler = RandomSampler(data)
            return train_data, DataLoader(train_data, sampler=train_sampler, batch_size=self.opt.train_batch_size), all_tran_indices, all_span_indices, extended_attention_mask, current_VDC
        else:
            eval_data = data
            eval_sampler = SequentialSampler(eval_data)
            return eval_data, DataLoader(eval_data, sampler=eval_sampler, batch_size=self.opt.eval_batch_size), all_tran_indices, all_span_indices, extended_attention_mask, current_VDC

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
            self._read_tsv(os.path.join(data_dir, "test.raw")), "dev")
    
    def get_validation_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "validation.raw")), "test")

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
