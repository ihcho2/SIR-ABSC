# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class BucketIterator_2(object):
    def __init__(self, data, batch_size, max_seq_length, other=None, sort_key='text_indices', shuffle=True, sort=False, 
                 input_format = None):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batch_size=batch_size
        self.data=[item for item in data if len(item['text_indices'])<110 ]
        self.max_seq_length = max_seq_length
        self.input_format = input_format
        
        dg_max_len = max([len(t['tran_indices']) for t in self.data])
        if other is not None:
            self.other=[item for item in other if len(item['text_indices'])<110 ]
            random.shuffle(self.other)
            self.batches = self.sort_and_pad(data, batch_size, self.other)
        else:
            self.other=None
            self.batches = self.sort_and_pad(self.data, batch_size, max_seq_length = self.max_seq_length, max_len = dg_max_len)
        
        self.batch_len = len(self.batches)
        
    def sort_and_pad(self, data, batch_size, max_seq_length, other=None, max_len=None):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = data
        if other is not None:
            num_k = int(math.ceil(len(other) / batch_size))    
            batches = []
            for i in range(num_batch):
                if i<num_k: k=i
                else:
                    k=random.randint(0,num_k-1)
                batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size],other[k*batch_size : (k+1)*batch_size], global_max_len1 = max_len))
        else:
            batches = []
            for i in range(num_batch):
                batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size], max_seq_length = max_seq_length, global_max_len1 = max_len))
        return batches

    def pad_data(self, batch_data, max_seq_length, other_data=None, global_max_len1 = None):
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
        if other_data is not None:
            max_len = max([len(t[self.sort_key]) for t in batch_data+other_data])
            max_len1 = max([len(t['tran_indices']) for t in batch_data+other_data])
        else:
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
            
            ########### Appending target at the end
            
            
            # GCLS
#             text_indices = [101] + text_indices
#             text_indices = [101] + text_indices + aspect_indices[1:]
#             text_indices = [101] + [30500] + text_indices[1:] + [30500] +\
#             tokenizer.convert_tokens_to_ids(tokenizer.tokenize('target is'))[1:] + aspect_indices[1:]
            
            if self.input_format == 'target_is':
                text_indices = [101] + [30500] + text_indices[1:] + [30500] +\
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize('target is'))[1:] + aspect_indices[1:]
                
            elif self.input_format == 'g_target_is':
                text_indices = [101] + [30500] + text_indices[1:] + [30500] +\
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize('target is')) + aspect_indices[1:]
                
            elif self.input_format == 'g_comma_target_is':
                text_indices = [101] + [30500] + text_indices[1:] + [30500] +\
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(', target is')) + aspect_indices[1:]
                
            elif self.input_format == 'target_is_g_comma_X':
                text_indices = [101] + [30500] + text_indices[1:] +\
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize('target is')) +[30500] +\
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(',')) + aspect_indices[1:]
                
            elif self.input_format == 'TD_X':
                text_indices_copy = text_indices.copy()
                text_indices = [101] + text_indices_copy[1:] + aspect_indices[1:]
                
            elif self.input_format == 'TD_no_X':
                text_indices_copy = text_indices.copy()
                text_indices = [101] + text_indices_copy[1:]
            
            elif self.input_format == 'X':
                text_indices_copy = text_indices.copy()
                text_indices = [101] + [30500] + text_indices_copy[1:] + aspect_indices[1:]
            
            elif self.input_format == 'g':
                text_indices_copy = text_indices.copy()
                text_indices = [101] + [30500] + text_indices_copy[1:] + [30500] + [102]
                
            elif self.input_format == 'no_add':
                text_indices_copy = text_indices.copy()
                text_indices = [101] + [30500] + text_indices_copy[1:]
            
            elif self.input_format == 'gX':
                text_indices_copy = text_indices.copy()
                sent = ' '
                for item in aspect:
                    sent += ' '
                    sent += item

                text_indices = [101] + [30500] + text_indices_copy[1:] + [30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                
            elif self.input_format == 'g_and_X':
                text_indices_copy = text_indices.copy()
                sent = 'and'
                for item in aspect:
                    sent += ' '
                    sent += item

                text_indices = [101] + [30500] + text_indices_copy[1:] + [30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                
            elif self.input_format == 'g_infront_X_g_and_X':
                text_indices_copy = text_indices.copy()
                sent = 'and'
                for item in aspect:
                    sent += ' '
                    sent += item
                
                assert len(span_indices) == 1
                
                x = tran_indices[span_indices[0][0]][0] + 1
                
                text_indices = text_indices_copy[:x] + [30500] + text_indices_copy[x:]+[30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                
            elif self.input_format == 'g_infront_X_X':
                text_indices_copy = text_indices.copy()
                sent = ''
                for item in aspect:
                    sent += ' '
                    sent += item
                
                assert len(span_indices) == 1
                
                x = tran_indices[span_indices[0][0]][0] + 1
                
                text_indices = text_indices_copy[:x] + [30500] + text_indices_copy[x:]+ tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                
            elif self.input_format == 'g_infront_X_gX':
                text_indices_copy = text_indices.copy()
                sent = ''
                for item in aspect:
                    sent += ' '
                    sent += item
                
                assert len(span_indices) == 1
                
                x = tran_indices[span_indices[0][0]][0] + 1
                
                text_indices = text_indices_copy[:x] + [30500] + text_indices_copy[x:]+[30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                
            elif self.input_format == 'g_infront_X_g':
                text_indices_copy = text_indices.copy()
                sent = ''
                for item in aspect:
                    sent += ' '
                    sent += item
                
                assert len(span_indices) == 1
                
                x = tran_indices[span_indices[0][0]][0] + 1
                
                text_indices = text_indices_copy[:x] + [30500] + text_indices_copy[x:]+[30500] + [102]
                
                
            elif self.input_format == 's_and_g_and_X':
                text_indices_copy = text_indices.copy()
                sent = 'and'
                sent2 = ' and '
                for item in aspect:
                    sent += ' '
                    sent += item

                text_indices = [101] + [30500] + text_indices_copy[1:] + [101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent2)) + [30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                
            elif self.input_format == 's2_gX':
                text_indices_copy = text_indices.copy()
                sent = ''
                for item in aspect:
                    sent += ' '
                    sent += item
                    
                text_indices = [101] + text_indices_copy[1:] + [30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
            
            elif self.input_format == 's2_g_and_X':
                text_indices_copy = text_indices.copy()
                sent = 'and'
                for item in aspect:
                    sent += ' '
                    sent += item
                    
                text_indices = [101] + text_indices_copy[1:] + [30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                
            elif self.input_format == 's_and_g':
                text_indices_copy = text_indices.copy()

                text_indices = [101] + [30500] + text_indices_copy[1:] + [101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize('and')) + [30500] + [102]
                
            elif self.input_format == 'based_on_s_and_g':
                text_indices_copy = text_indices.copy()

                text_indices = [101] + [30500] + text_indices_copy[1:] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize('based on ')) + [101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize('and')) + [30500] + [102]
                
            elif self.input_format == 's_g_and_X':
                text_indices_copy = text_indices.copy()
                sent = ', and'
                for item in aspect:
                    sent += ' '
                    sent += item

                text_indices = [101] + [30500] + text_indices_copy[1:] + [101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(',')) + [30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                
            elif self.input_format == 's_g_X':
                text_indices_copy = text_indices.copy()
                sent = ''
                for item in aspect:
                    sent += ' '
                    sent += item

                text_indices = [101] + [30500] + text_indices_copy[1:] + [101] + [30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                
            elif self.input_format == 'what_do_you_think_1':
                text_indices_copy = text_indices.copy()
                sent = 'what do you think of the'
                for item in aspect:
                    sent += ' '
                    sent += item
                    
                sent += ' of it ?'

                text_indices = [101] + text_indices_copy[1:] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                
            elif self.input_format == 'what_do_you_think_2':
                text_indices_copy = text_indices.copy()
                sent = 'what do you think of'
                for item in aspect:
                    sent += ' '
                    sent += item
                    
                sent += ' using'

                text_indices = [101] + [30500] + text_indices_copy[1:] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize('and')) + [30500]+[102]
                
            elif self.input_format == 'what_do_you_think_3':
                text_indices_copy = text_indices.copy()
                sent = 'what do you think of'
                for item in aspect:
                    sent += ' '
                    sent += item
                    
                sent += 'of it based on'

                text_indices = [101] + [30500] + text_indices_copy[1:] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize('and')) + [30500]+[102]
                
            elif self.input_format == 'what_do_you_think_4':
                text_indices_copy = text_indices.copy()
                sent = 'what do you think of'
                for item in aspect:
                    sent += ' '
                    sent += item
                    
                sent += ' based on'

                text_indices = [101] + [30500] + text_indices_copy[1:] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize('and')) + [30500]+[102]
                
            elif self.input_format == 's_g':
                text_indices_copy = text_indices.copy()
                sent = ''
                for item in aspect:
                    sent += ' '
                    sent += item

                text_indices = [101] + [30500] + text_indices_copy[1:] + [101] + [30500] + [102]
                
            elif self.input_format == 's_comma_g':
                text_indices_copy = text_indices.copy()
                sent = ''
                for item in aspect:
                    sent += ' '
                    sent += item

                text_indices = [101] + [30500] + text_indices_copy[1:] + [101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(',')) + [30500] + [102]
                
            elif self.input_format == 'g_comma_X':
                text_indices_copy = text_indices.copy()
                sent = ','
                for item in aspect:
                    sent += ' '
                    sent += item

                text_indices = [101] + [30500] + text_indices_copy[1:] + [30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                
            elif self.input_format == 'gX_2':
                text_indices_copy = text_indices.copy()
                sent = ''
                for item in aspect:
                    sent += ' '
                    sent += item

                text_indices = [101] + [30500] + text_indices_copy[1:] + [30500] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [102]
                
            elif self.input_format == 'Xg':
                text_indices_copy = text_indices.copy()
                sent = ''
                for item in aspect:
                    sent += ' '
                    sent += item

                text_indices = [101] + [30500] + text_indices_copy[1:] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) + [30500] + [102]
            
#             text_indices = text_indices + aspect_indices[1:] \
#             + tokenizer.convert_tokens_to_ids(tokenizer.tokenize('target is')) + aspect_indices[1:] \
#             + tokenizer.convert_tokens_to_ids(tokenizer.tokenize('aspect is ')) + aspect_indices[1:] \
#             + tokenizer.convert_tokens_to_ids(tokenizer.tokenize('what do you think of ')) + aspect_indices[1:]
            
            # fix the segment_ids in utils/data_util.py
            ###########
            
            ########### Appending target at the beginning
            # text_indices = aspect_indices + text_indices[1:]
            # fix the segment_ids in utils/data_util.py
            ###########
            
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
        if other_data is not None:
            for item in other_data:
                text_indices, context_indices, span_indices,tran_indices,aspect_indices, left_indices, polarity, dependency_graph,text,aspect = \
                    item['text_indices'], item['context_indices'],item['span_indices'],item['tran_indices'], item['aspect_indices'], item['left_indices'],\
                    item['polarity'], item['dependency_graph'],item['text'], item['aspect']
                text_padding = [0] * (max_len - len(text_indices))
                batch_text_indices.append(text_indices + text_padding)
                batch_text.append(text)
                batch_polarity.append(polarity)
                batch_tran.append(tran_indices)
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
