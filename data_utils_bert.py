# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from tqdm import tqdm
import re
from transformers import BertTokenizer
import argparse
import spacy

# from transformers.optimization import AdamW, WarmupLinearSchedule
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer.
#bert_model= BertModel.from_pretrained('bert-base-uncased')
    
class Tokenizer(object):
    def __init__(self, word2idx=None,tokenizer=None):
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.do_basic_tokenize=False
        print('load successfully')
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower().strip()
        words = tokenize(text)
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, tran=False):
        text = text.lower().strip()
        words = tokenize(text)  # This is the parser's tokenizer.
        trans=[]
        realwords=[]
        
        for word in words:
            wordpieces=self.tokenizer.tokenize(word)  # Tokenize again with BERT's tokenizer.
            tmplen=len(realwords)
            
            realwords.extend(wordpieces)
            trans.append([tmplen,len(realwords)])
            
            assert tmplen != len(realwords)
                
        sequence = [self.tokenizer._convert_token_to_id('[CLS]')]+[self.tokenizer._convert_token_to_id(w) for w in realwords]+[self.tokenizer._convert_token_to_id('[SEP]')]
        if len(sequence) == 0:
            sequence = [0]
        if tran: return sequence,trans
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def span(texts,aspect):
    startid=0
    aslen=len(tokenize(aspect))
    spans=[]
    
    for idx,text in enumerate(texts):
        tmp=len(tokenize(text))
        startid+=tmp
        tmp=startid
        if idx < len(texts)-1:
            startid+=aslen
            spans.append([tmp,startid])
            
            assert tmp != startid
            
    return spans

        
class ABSADatesetReader:
    @staticmethod
    def __read_data__(fname, tokenizer, parser_info=None):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname + '_' + parser_info + '.graph', 'rb')
        fin1 = open('./datasets/' + parser_info + '.global_edge_vocab', 'rb')
        fin2 = open('./datasets/' + parser_info + '.global_pos_vocab', 'rb')
        
        idx2gragh = pickle.load(fin)
        edgevocab=pickle.load(fin1)
        posvocab=pickle.load(fin2)
        fin1.close()
        fin2.close()
        fin.close()


        all_data = []
        for i in tqdm(range(0, len(lines), 3)):
            text_left = [s.lower().strip() for s in lines[i].split("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            
            span_indices=span(text_left,aspect)
            assert len(span_indices)>=1
            concats=concat(text_left,aspect)
            
            text_indices, tran_indices = tokenizer.text_to_sequence(concats,True)
            context_indices = tokenizer.text_to_sequence(concats)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(concats)
            
            polarity = int(polarity)+1
            
            dependency_graph = idx2gragh[i]
            
            assert len(idx2gragh[i][0])==len(tokenize(concats))
            
            data = {
                'text': tokenize(concats.lower().strip()),
                'aspect': tokenize(aspect),
                'text_indices': text_indices,
                'tran_indices': tran_indices,
                'context_indices': context_indices,
                'span_indices': span_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
            }

            all_data.append(data)
        return all_data
    
    def __init__(self, dataset='rest14', parser_info = None):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurants/restaurant_train.raw',
                'test': './datasets/semeval14/restaurants/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptops/laptop_train.raw',
                'test': './datasets/semeval14/laptops/laptop_test.raw'
            },
            'mams': {
                'train': './datasets/MAMS-ATSA/train.raw',
                'test': './datasets/MAMS-ATSA/test.raw',
                'validation': './datasets/MAMS-ATSA/validation.raw'
            }
        }
        
        self.tokenizer = Tokenizer()
        
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], self.tokenizer,
                                                                      parser_info = parser_info))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], self.tokenizer,
                                                                     parser_info = parser_info))
        if dataset== 'mams':
            self.validation_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['validation'], self.tokenizer,
                                                                     parser_info = parser_info))
        
            

def get_config():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--parser_info",
                        default='spacy_sm_3.3.0',
                        type=str,
                        help="Parser you want to use.")
    
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_config()
    
    print('='*77)
    print('args.parser_info: ', args.parser_info)
    print('='*77)
    
    if 'spacy_sm' in args.parser_info:
        nlp = spacy.load('en_core_web_sm')
    elif 'spacy_lg' in args.parser_info:
        nlp = spacy.load('en_core_web_lg')
        
    def tokenize(text):
        text=text.strip()
        text=re.sub(r' {2,}',' ',text)
        document = nlp(text)

        return [token.text for token in document]
    
    def concat(texts,aspect):
        source=''
        splitnum=0

        for i,text in enumerate(texts):
            source+=text
            splitnum+=len(tokenize(text))
            if i <len(texts)-1:
                source+=' '+aspect+' '
                splitnum+=len(tokenize(aspect))

        if splitnum!=len(tokenize(source.strip())):
            print('ERROR')

        return re.sub(r' {2,}',' ',source.strip())
    
    # 1. Laptop
    tmp=ABSADatesetReader(dataset='lap14', parser_info = args.parser_info)
    dataset='lap14'
    with open(dataset+'_datas_bert_' + args.parser_info + '.pkl', 'wb') as f:
        pickle.dump(tmp, f)
    
    # 2. Restaurant
    tmp=ABSADatesetReader(dataset='rest14', parser_info = args.parser_info)
    dataset='rest14'
    with open(dataset+'_datas_bert_' + args.parser_info + '.pkl', 'wb') as f:
        pickle.dump(tmp, f)
            
    # 3. Twitter
    tmp=ABSADatesetReader(dataset='twitter', parser_info = args.parser_info)
    dataset='twitter'
    with open(dataset+'_datas_bert_' + args.parser_info + '.pkl', 'wb') as f:
        pickle.dump(tmp, f)
            
    # 4. MAMS
    tmp=ABSADatesetReader(dataset='mams', parser_info = args.parser_info)
    dataset='mams'
    with open(dataset+'_datas_bert_' + args.parser_info + '.pkl', 'wb') as f:
        pickle.dump(tmp, f)
