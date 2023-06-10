# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
import tqdm
import re
import argparse


def tokenize(text):
    text=text.strip()
    text=re.sub(r' {2,}',' ',text)
    document = nlp(text)
    
    return [token.text for token in document]


def update_vocabs(text, edge_vocab, pos_vocab):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    for token in document:
        if token.dep_ not in edge_vocab:
            edge_vocab[token.dep_] = len(edge_vocab)
        if token.pos_ not in pos_vocab:
            pos_vocab[token.pos_] = len(pos_vocab)
            
    return edge_vocab, pos_vocab


def dependency_adj_matrix(text, edge_vocab, pos_vocab):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text.strip())
    seq_len = len(tokenize(text))
    assert len(document)==seq_len
    
    # Creating adjacency matrices, edge_matrices, and POS tags.
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    matrix1 = np.zeros((seq_len, seq_len)).astype('float32')
    edge = np.zeros((seq_len, seq_len)).astype('int32')
    edge1 = np.zeros((seq_len, seq_len)).astype('int32')
    pos_tag = np.zeros(seq_len).astype('int32')
    
    for token in document:
        if token.i >= seq_len:
            print('ERROR')
            
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            matrix1[token.i][token.i] = 1
            
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix1[child.i][token.i] = 1
                    edge[token.i][child.i] = edge_vocab.get(child.dep_,1)
                    edge1[child.i][token.i] = edge_vocab.get(child.dep_,1)
                    pos_tag[token.i] = pos_vocab.get(token.pos_,1)
                    
    return matrix, matrix1, edge, edge1, pos_tag


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


def process(filename, edge_vocab = None, pos_vocab = None, savevocab = True, parser_info = None):
    if edge_vocab is not None:
        pass
    else:
        edge_vocab={'<pad>': 0, '<unk>': 1}
        pos_vocab={'<pad>':0, '<unk>':1}
        
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    
    fout = open(filename+'_'+parser_info+'.graph', 'wb')
    
    if savevocab:
        fout1 = open(filename+'_'+parser_info+'.edgevocab', 'wb')
        fout2 = open(filename+'_'+parser_info+'.posvocab', 'wb')
        
    if savevocab:
        for i in tqdm.tqdm(range(0, len(lines), 3)):
            text_left = [s.lower().strip() for s in lines[i].split("$T$")]
            aspect = lines[i + 1].lower().strip()
            
            edge_vocab, pos_vocab = update_vocabs(concat(text_left,aspect), edge_vocab, pos_vocab)
            
    for i in tqdm.tqdm(range(0, len(lines), 3)):
        text_left = [s.lower().strip() for s in lines[i].split("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(concat(text_left,aspect), edge_vocab, pos_vocab)
        idx2graph[i] = adj_matrix
        
    pickle.dump(idx2graph, fout) 
    
    if savevocab:
        pickle.dump(edge_vocab, fout1)
        pickle.dump(pos_vocab, fout2)
        
    fout.close() 
    
    if savevocab:
        fout1.close() 
        fout2.close()
        
    return edge_vocab, pos_vocab

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

    # 1. Laptop
    edge_vocab, pos_vocab =process('./datasets/semeval14/laptops/laptop_train.raw', None, True, 
                                   parser_info=args.parser_info)
    process('./datasets/semeval14/laptops/laptop_test.raw', edge_vocab, pos_vocab, False, parser_info=args.parser_info)
    
    
    # 2. Restaurant
    edge_vocab, pos_vocab =process('./datasets/semeval14/restaurants/restaurant_train.raw', None, True, 
                                   parser_info = args.parser_info)
    process('./datasets/semeval14/restaurants/restaurant_test.raw', edge_vocab, pos_vocab, False, 
            parser_info=args.parser_info)
    
    
    # 3. Twitter
    edge_vocab, pos_vocab =process('./datasets/acl-14-short-data/train.raw', None, True, parser_info=args.parser_info)
    process('./datasets/acl-14-short-data/test.raw', edge_vocab, pos_vocab, False, parser_info=args.parser_info)

    
    # 4. MAMS
    edge_vocab, pos_vocab =process('./datasets/MAMS-ATSA/train.raw', None, True, parser_info=args.parser_info)
    process('./datasets/MAMS-ATSA/test.raw', edge_vocab, pos_vocab, False, parser_info=args.parser_info)
    process('./datasets/MAMS-ATSA/validation.raw', edge_vocab, pos_vocab, False, parser_info=args.parser_info)
