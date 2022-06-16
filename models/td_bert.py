import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modeling import BertModel


# CrossEntropyLoss for Label Smoothing Regularization
class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        # one_hot_label = torch.zeros(batchsize, classes) + prob
        one_hot_label = torch.zeros(batchsize, classes)
        for i in range(batchsize):
            index = label[i]
            # one_hot_label[i, :] += prob * 2
            # one_hot_label[i, index] += (1.0 - self.para_LSR)
            if index != 1:  # If neutral, smooth; otherwise, not smooth
                one_hot_label[i, :] += prob
                one_hot_label[i, index] += (1.0 - self.para_LSR)
            else:
                one_hot_label[i, index] = 1
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
        loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class TD_BERT(nn.Module):
    def __init__(self, config, opt):
        super(TD_BERT, self).__init__()
        self.opt = opt
        n_filters = opt.n_filters  # The number of convolution kernels
        filter_sizes = opt.filter_sizes  # Convolution kernel size, multiple sizes are passed as a list
        embedding_dim = opt.embed_dim  
        output_dim = opt.output_dim  # Output dimension, here is 3, representing negative, neutral, positive respectively
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(opt.keep_dropout)
        self.fc = nn.Linear(embedding_dim, output_dim)  # fully connected layer bbfc
        # self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)  # fully connected layer tc_cnn
        # self.bn1 = nn.BatchNorm1d(output_dim)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                segment_t_ids=None, input_left_ids=None, input_left_mask=None, segment_left_ids=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sentence_embed = all_encoder_layers[-1]  # Use the last layer of encoding results for classification
        # sentence_embed = sum(all_encoder_layers)  # All layers are superimposed, I have tried it, the effect is not good, the convergence is slow, and the upper limit is low
        
        target_in_sent_embed = torch.zeros(input_ids.size()[0], sentence_embed.size()[-1]).to(
            self.opt.device)  # The embedding vector of the target word in the sentence
        left_len = torch.sum(input_left_ids != 0, dim=-1) - 1  # Note that there are [CLS] and [SEP] at the beginning and end
        target_len = torch.sum(input_t_ids != 0, dim=1) - 2  # Note that there are [CLS] and [SEP] at the beginning and end
        target_in_sent_idx = torch.cat([left_len.unsqueeze(-1), (left_len + target_len).unsqueeze(-1)], dim=-1)

        for i in range(input_ids.size()[0]):  # iterate over each of the batches
            target_embed = sentence_embed[i][target_in_sent_idx[i][0]:target_in_sent_idx[i][1]]  # batch_size * max_seq_len * embedding_dim
            target_in_sent_embed[i] = torch.max(target_embed, dim=0)[0]  # Converted to 1 * embedding_dim, taking the maximum value is the best (max_pooling)
            # target_in_sent_embed[i] = target_embed.sum(dim=0)  # sum
            # target_in_sent_embed[i] = torch.mean(target_embed, dim=0)[0]  # average

        cat = self.dropout(target_in_sent_embed)
        logits = self.fc(cat)
        logits = torch.tanh(logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # Standard cross-entropy loss function
            # loss_fct = CrossEntropyLoss_LSR(device=self.opt.device, para_LSR=self.opt.para_LSR)  # Loss function after label smoothing，para_LSR The interval is 0.1~0.9
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
        
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.dropout = nn.Dropout(0.1)
        
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        stdv = 1. / math.sqrt(self.bias.shape[0])
        torch.nn.init.uniform_(self.bias, a=-stdv, b=stdv)
            
    def forward(self, text, adj1, adj2):
        adj = adj1 + adj2
        adj[adj>=1]=1
        
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True)+0.0000001
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
class BiGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(BiGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight3 = nn.Parameter(torch.FloatTensor(2*out_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
            self.bias2 = nn.Parameter(torch.FloatTensor(out_features))
            self.bias3 = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias2', None)
            self.register_parameter('bias3', None)
            
        self.dropout = nn.Dropout(0.1)
        
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight3)
        stdv = 1. / math.sqrt(self.bias.shape[0])
        torch.nn.init.uniform_(self.bias, a=-stdv, b=stdv)
        stdv = 1. / math.sqrt(self.bias2.shape[0])
        torch.nn.init.uniform_(self.bias2, a=-stdv, b=stdv)
        stdv = 1. / math.sqrt(self.bias3.shape[0])
        torch.nn.init.uniform_(self.bias3, a=-stdv, b=stdv)
            
    def forward(self, text, adj1, adj2):
        hidden = torch.matmul(text, self.weight)
        hidden2 = torch.matmul(text, self.weight2)
        denom1 = torch.sum(adj1, dim=2, keepdim=True)+1
        denom2 = torch.sum(adj2, dim=2, keepdim=True)+1
        
        output = F.relu(torch.matmul(adj1, hidden) / denom1 + self.bias)
        output2 = F.relu(torch.matmul(adj2, hidden2) /denom2 + self.bias2)
        
        output3 = torch.matmul(torch.cat((output,output2), dim=-1), self.weight3) + self.bias3
        
        return output3
        
        
class TD_BERT_with_GCN(nn.Module):
    def __init__(self, config, opt):
        super(TD_BERT_with_GCN, self).__init__()
        self.opt = opt
        n_filters = opt.n_filters  # The number of convolution kernels
        filter_sizes = opt.filter_sizes  # Convolution kernel size, multiple sizes are passed as a list
        embedding_dim = opt.embed_dim  
        output_dim = opt.output_dim  # Output dimension, here is 3, representing negative, neutral, positive respectively
        self.bert = BertModel(config)
        if opt.bigcn == str(True):
            print('-'*77)
            print('Using BiGCN')
            print('-'*77)
            self.gcn1 = BiGraphConvolution(opt.gcn_hidden_dim, opt.gcn_hidden_dim)
            self.gcn2 = BiGraphConvolution(opt.gcn_hidden_dim, opt.gcn_hidden_dim)
            self.gcn3 = BiGraphConvolution(opt.gcn_hidden_dim, opt.gcn_hidden_dim)
        else:
            print('-'*77)
            print('Using GCN')
            print('-'*77)
            self.gcn1 = GraphConvolution(opt.gcn_hidden_dim, opt.gcn_hidden_dim)
            self.gcn2 = GraphConvolution(opt.gcn_hidden_dim, opt.gcn_hidden_dim)
            self.gcn3 = GraphConvolution(opt.gcn_hidden_dim, opt.gcn_hidden_dim)
            
        self.dropout = nn.Dropout(opt.keep_dropout)
        self.fc = nn.Linear(embedding_dim+opt.gcn_hidden_dim, output_dim)
        self.weight = nn.Parameter(torch.FloatTensor(768, opt.gcn_hidden_dim))
        
        self.init_parameters()
    
    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                segment_t_ids=None, input_left_ids=None, input_left_mask=None, segment_left_ids=None, dg= None, dg1=None, tran_indices=None, span_indices=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sentence_embed = all_encoder_layers[-1]  # Use the last layer of encoding results for classification
        # sentence_embed = sum(all_encoder_layers)  # All layers are superimposed, I have tried it, the effect is not good, the convergence is slow, and the upper limit is low
        
                
        target_in_sent_embed = torch.zeros(input_ids.size()[0], sentence_embed.size()[-1]).to(
            self.opt.device)  # The embedding vector of the target word in the sentence
        left_len = torch.sum(input_left_ids != 0, dim=-1) - 1  # Note that there are [CLS] and [SEP] at the beginning and end
        target_len = torch.sum(input_t_ids != 0, dim=1) - 2  # Note that there are [CLS] and [SEP] at the beginning and end
        target_in_sent_idx = torch.cat([left_len.unsqueeze(-1), (left_len + target_len).unsqueeze(-1)], dim=-1)

        for i in range(input_ids.size()[0]):  # iterate over each of the batches
            target_embed = sentence_embed[i][target_in_sent_idx[i][0]:target_in_sent_idx[i][1]]  # batch_size * max_seq_len * embedding_dim
            target_in_sent_embed[i] = torch.max(target_embed, dim=0)[0]  # Converted to 1 * embedding_dim, taking the maximum value is the best (max_pooling)
            # target_in_sent_embed[i] = target_embed.sum(dim=0)  # sum
            # target_in_sent_embed[i] = torch.mean(target_embed, dim=0)[0]  # average

        cat = self.dropout(target_in_sent_embed)
        
        # 먼저 BERT tokenized => DG 크기로 수정해야 함. DGEDT를 따라 그냥 sum을 사용.
        tmps=torch.zeros(input_ids.size()[0], dg[0].size(-1), 768).float().to(self.opt.device)
        for i,spans in enumerate(tran_indices):
            for j,span in enumerate(spans):
                tmps[i,j]=torch.sum(sentence_embed[i,span[0]+1:span[1]+1],0)
                
        gcn_text = self.dropout(tmps)
        gcn_text = F.relu(torch.matmul(gcn_text, self.weight))
        gcn_text=self.dropout(F.relu(self.gcn1(gcn_text, dg, dg1))) + gcn_text
        gcn_text=self.dropout(F.relu(self.gcn2(gcn_text, dg, dg1))) + gcn_text
        gcn_text=self.dropout(F.relu(self.gcn3(gcn_text, dg, dg1))) + gcn_text
        
        gcn_target_embed = torch.zeros(input_ids.size()[0], gcn_text.size()[-1]).float().to(self.opt.device)
        for i in range(input_ids.size()[0]):
            x = gcn_text[i, span_indices[i][0][0]:span_indices[i][0][1]]
            gcn_target_embed[i] = torch.sum(x, 0)
            
        gcn_cat = self.dropout(gcn_target_embed)
                             
        # Concatenate the output of BERT and GCN.
        concatenated_embed = torch.cat((cat, gcn_cat), dim=1)
        
        logits = self.fc(concatenated_embed)
        logits = torch.tanh(logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # Standard cross-entropy loss function
            # loss_fct = CrossEntropyLoss_LSR(device=self.opt.device, para_LSR=self.opt.para_LSR)  # Loss function after label smoothing，para_LSR The interval is 0.1~0.9
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
