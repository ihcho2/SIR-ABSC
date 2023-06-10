import pickle
import argparse

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
    
    # Specify the edge, pos vocabs for each dataset (Laptop, Restaurant, Twitter, MAMS)
    
    fname = {
            'twitter': './datasets/acl-14-short-data/train.raw',
            'rest14': './datasets/semeval14/restaurants/restaurant_train.raw',
            'lap14': './datasets/semeval14/laptops/laptop_train.raw',
            'mams': './datasets/MAMS-ATSA/train.raw'
            }
    
    edge_vocab_L = open(fname['lap14']+'_'+parser_info+'.edgevocab', 'wb')
    edge_vocab_R = open(fname['rest14']+'_'+parser_info+'.edgevocab', 'wb')
    edge_vocab_T = open(fname['twitter']+'_'+parser_info+'.edgevocab', 'wb')
    edge_vocab_M = open(fname['mams']+'_'+parser_info+'.edgevocab', 'wb')

    pos_vocab_L = open(fname['lap14']+'_'+parser_info+'.posvocab', 'wb')
    pos_vocab_R = open(fname['reset14']+'_'+parser_info+'.posvocab', 'wb')
    pos_vocab_T = open(fname['twitter']+'_'+parser_info+'.posvocab', 'wb')
    pos_vocab_M = open(fname['mams']+'_'+parser_info+'.posvocab', 'wb')
    
    global_edge_vocab = {}
    global_pos_vocab = {}
    
    
    # global_edge_vocab
    for key in edge_vocab_L.keys():
        if key not in global_edge_vocab.keys():
            global_edge_vocab[key] = len(global_edge_vocab)
            
    for key in edge_vocab_R.keys():
        if key not in global_edge_vocab.keys():
            global_edge_vocab[key] = len(global_edge_vocab)
            
    for key in edge_vocab_T.keys():
        if key not in global_edge_vocab.keys():
            global_edge_vocab[key] = len(global_edge_vocab)
            
    for key in edge_vocab_M.keys():
        if key not in global_edge_vocab.keys():
            global_edge_vocab[key] = len(global_edge_vocab)
            
            
    # global_pos_vocab 
    for key in pos_vocab_L.keys():
        if key not in global_pos_vocab.keys():
            global_pos_vocab[key] = len(global_pos_vocab)
            
    for key in pos_vocab_R.keys():
        if key not in global_pos_vocab.keys():
            global_pos_vocab[key] = len(global_pos_vocab)
            
    for key in pos_vocab_T.keys():
        if key not in global_pos_vocab.keys():
            global_pos_vocab[key] = len(global_pos_vocab)
            
    for key in pos_vocab_M.keys():
        if key not in global_pos_vocab.keys():
            global_pos_vocab[key] = len(global_pos_vocab)
            
    # print stats
    print('-'*77)
    print('len(edge_vocab_L): ', len(edge_vocab_L))
    print('len(edge_vocab_R): ', len(edge_vocab_R))
    print('len(edge_vocab_T): ', len(edge_vocab_T))
    print('len(edge_vocab_M): ', len(edge_vocab_M))
    print('len(global_edge_vocab): ', len(global_edge_vocab))
    print('-'*77)
    print('len(pos_vocab_L): ', len(pos_vocab_L))
    print('len(pos_vocab_R): ', len(pos_vocab_R))
    print('len(pos_vocab_T): ', len(pos_vocab_T))
    print('len(pos_vocab_M): ', len(pos_vocab_M))
    print('len(global_pos_vocab): ', len(global_pos_vocab))
    print('-'*77)
    
    fout_edge = open(filename+'_'+parser_info+'.global_edge_vocab', 'wb')
    fout_pos = open(filename+'_'+parser_info+'.global_pos_vocab', 'wb')
    
    pickle.dump(global_edge_vocab, fout_edge)
    pickle.dump(global_pos_vocab, fout_pos)