# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy
from transformers import RobertaTokenizer

class BucketIterator(object):
    def __init__(self, data, batch_size, max_seq_length, sort_key='text_indices', shuffle=True, sort=False,
                 input_format = None, model_name = None):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batch_size=batch_size
        self.input_format = input_format
        self.model_name = model_name
        
        if 'bert_' in model_name:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif 'roberta' in model_name:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        self.data=[item for item in data if len(item['text_indices'])<110 ]
        print('len(data): ', len(data))
        print('len(self.data): ', len(self.data))
        self.max_seq_length = max_seq_length
        
        dg_max_len = max([len(t['tran_indices']) for t in self.data])
        
        self.batches = self.sort_and_pad(self.data, batch_size, max_seq_length = self.max_seq_length, max_len = dg_max_len)
        
        self.batch_len = len(self.batches)
        
    def sort_and_pad(self, data, batch_size, max_seq_length, max_len=None):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size], max_seq_length = max_seq_length, global_max_len1 = max_len))
        return batches

    def pad_data(self, batch_data, max_seq_length, global_max_len1 = None):
        batch_text_indices = []
        
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_dependency_graph1 = []
        batch_dependency_graph2 = []
        batch_dependency_graph3 = []
        batch_span=[]
        batch_tran=[]
        batch_text=[]
        batch_aspect=[]
        
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        max_len1 = max([len(t['tran_indices']) for t in batch_data])
        
        #################
        if global_max_len1 is not None:
            max_len1 = global_max_len1
        max_len = max_seq_length
        #################
        for item in batch_data:
            text_indices, context_indices, span_indices,tran_indices,aspect_indices, left_indices, polarity, dependency_graph,text,aspect = \
                item['text_indices'], item['context_indices'],item['span_indices'],item['tran_indices'], item['aspect_indices'], item['left_indices'],\
                item['polarity'], item['dependency_graph'],item['text'], item['aspect']
            
            ########### Modifying the input based on self.input_format
            # RoBERTa-TD
            if self.input_format == 'td':
                pass
            elif self.input_format == 'td_X':
                text_indices_copy = text_indices.copy()
                # for roberta_td
                text_indices = text_indices  + [2] + aspect_indices[1:]
                
            elif self.input_format == 'X':
                text_indices_copy = text_indices.copy()
                # for roberta_td
#                 text_indices = text_indices  + [2] + aspect_indices[1:]
                
                # for roberta_gcls
                if 'bert_' in self.model_name:
                    text_indices = [101] + [30500] + text_indices_copy[1:] + aspect_indices[1:]
                elif 'roberta' in self.model_name:
                    text_indices = [0] + [50249] + text_indices_copy[1:] + [2] + aspect_indices[1:]
            
            elif self.input_format == 'g':
                text_indices_copy = text_indices.copy()
                
                if 'bert_' in self.model_name:
                    text_indices = [101] + [30500] + text_indices_copy[1:] + [30500] + [102]
                elif 'roberta' in self.model_name:
                    text_indices = [0] + [50249] + text_indices_copy[1:] + [2] + [50249] + [2]
            
            elif self.input_format == 'gX':
                text_indices_copy = text_indices.copy()
                sent = ' '
                for item in aspect:
                    sent += ' '
                    sent += item
                
                if 'bert_' in self.model_name:
                    text_indices = [101] + [30500] + text_indices_copy[1:] + [30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                elif 'roberta' in self.model_name:
                    text_indices = [0] + [50249] + text_indices_copy[1:] + [2] + [50249] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent)) + [2]
                
            elif self.input_format == 'g_infront_of_X_g_and_X':
                text_indices_copy = text_indices.copy()
                sent = 'and'
                for item in aspect:
                    sent += ' '
                    sent += item
                
                assert len(span_indices) == 1
                
                x = tran_indices[span_indices[0][0]][0] + 1
                
                if 'bert_' in self.model_name:
                    text_indices = text_indices_copy[:x] + [30500] + text_indices_copy[x:]+[30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                elif 'roberta' in self.model_name:
                    text_indices = text_indices_copy[:x] + [50249] + text_indices_copy[x:]+ [2] + [50249] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [2]
                    
            elif self.input_format == 'g_infront_of_X_gX':
                text_indices_copy = text_indices.copy()
                sent = ''
                for item in aspect:
                    sent += ' '
                    sent += item
                
                assert len(span_indices) == 1
                
                x = tran_indices[span_indices[0][0]][0] + 1
                
                if 'bert_' in self.model_name:
                    text_indices = text_indices_copy[:x] + [30500] + text_indices_copy[x:]+[30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                elif 'roberta' in self.model_name:
                    text_indices = text_indices_copy[:x] + [50249] + text_indices_copy[x:]+[2]+[50249] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [2]
                    
            elif self.input_format == 'g_infront_of_X_g':
                text_indices_copy = text_indices.copy()
                sent = ''
                for item in aspect:
                    sent += ' '
                    sent += item
                
#                 assert len(span_indices) == 1
                
                x = tran_indices[span_indices[0][0]][0] + 1
                
                if 'bert_' in self.model_name:
                    text_indices = text_indices_copy[:x] + [30500] + text_indices_copy[x:]+[30500] + [102]
                elif 'roberta' in self.model_name:
                    text_indices = text_indices_copy[:x] + [50249] + text_indices_copy[x:]+[2]+[50249] + [2]
                    
                    
            # pad = 1 for RoBERTa ? 
            if 'bert_' in self.model_name:
                text_padding = [0] * (max_len - len(text_indices))
                context_padding = [0] * (max_len - len(context_indices))
                aspect_padding = [0] * (max_len - len(aspect_indices))
                left_padding = [0] * (max_len - len(left_indices))
                batch_span.append(span_indices)
                batch_text_indices.append(text_indices + text_padding)
                batch_context_indices.append(context_indices + context_padding)
                batch_aspect_indices.append(aspect_indices + aspect_padding)
                batch_left_indices.append(left_indices + left_padding)
                batch_polarity.append(polarity)
                batch_tran.append(tran_indices)
                batch_dependency_graph.append(numpy.pad(dependency_graph[0], \
                    ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
                batch_dependency_graph1.append(numpy.pad(dependency_graph[1], \
                    ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
                batch_dependency_graph2.append(numpy.pad(dependency_graph[2], \
                    ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
                batch_dependency_graph3.append(numpy.pad(dependency_graph[3], \
                    ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
                batch_text.append(text)
                batch_aspect.append(aspect)
                
            elif 'roberta' in self.model_name:
                text_padding = [1] * (max_len - len(text_indices))

                context_padding = [1] * (max_len - len(context_indices))
                aspect_padding = [1] * (max_len - len(aspect_indices))
                left_padding = [1] * (max_len - len(left_indices))
                batch_span.append(span_indices)
                batch_text_indices.append(text_indices + text_padding)

                batch_context_indices.append(context_indices + context_padding)
                batch_aspect_indices.append(aspect_indices + aspect_padding)
                batch_left_indices.append(left_indices + left_padding)
                batch_polarity.append(polarity)
                batch_tran.append(tran_indices)
                batch_dependency_graph.append(numpy.pad(dependency_graph[0], \
                    ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
                batch_dependency_graph1.append(numpy.pad(dependency_graph[1], \
                    ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
                batch_dependency_graph2.append(numpy.pad(dependency_graph[2], \
                    ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
                batch_dependency_graph3.append(numpy.pad(dependency_graph[3], \
                    ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
                batch_text.append(text)
                batch_aspect.append(aspect)
                
                
        return { \
                'text_indices': torch.tensor(batch_text_indices), \
                'text':batch_text,\
                'aspect':batch_aspect,\
                'context_indices': torch.tensor(batch_context_indices), \
                'span_indices':batch_span,\
                'tran_indices':batch_tran,\
                'aspect_indices': torch.tensor(batch_aspect_indices), \
                'left_indices': torch.tensor(batch_left_indices), \
                'polarity': torch.tensor(batch_polarity), \
                'dependency_graph': torch.tensor(batch_dependency_graph),\
                'dependency_graph1': torch.tensor(batch_dependency_graph1),\
                'dependency_graph2': torch.tensor(batch_dependency_graph2).long(),\
                'dependency_graph3': torch.tensor(batch_dependency_graph3).long()
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)
            self.batches = self.sort_and_pad(self.data, self.batch_size)
        for idx in range(self.batch_len):
#            print(len(self.batches[idx]['text_indices']))
            yield self.batches[idx]
