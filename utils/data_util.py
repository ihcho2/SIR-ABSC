import tokenization_word as tokenization
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from data_utils import *
from bucket_iterator import BucketIterator
from bucket_iterator_2 import BucketIterator_2
import pickle
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
        elif opt.task_name == 'twitter':
            dgedt_dataset = 'twitter'
            
        absa_dataset=pickle.load(open(dgedt_dataset+'_datas.pkl', 'rb'))
        opt.edge_size=len(absa_dataset.edgevocab)
        self.train_data_loader = BucketIterator_2(data=absa_dataset.train_data, batch_size=100000, max_seq_length = self.opt.max_seq_length, shuffle=True)
        self.test_data_loader = BucketIterator_2(data=absa_dataset.test_data, batch_size=100000, max_seq_length = self.opt.max_seq_length, shuffle=False)
        
        self.DGEDT_train_data = self.train_data_loader.data
        self.DGEDT_train_batches = self.train_data_loader.batches
        
        self.DGEDT_test_data = self.test_data_loader.data
        self.DGEDT_test_batches = self.test_data_loader.batches
        
        if opt.model_name in ['gcls', 'scls', 'gcls_er', 'gcls_moe', 'gcls_moe_default']:
            self.train_gcls_attention_mask = self.process_DG(self.DGEDT_train_data, opt.gcls_length)
            self.eval_gcls_attention_mask = self.process_DG(self.DGEDT_test_data, opt.gcls_length)
        
        ######################
        
        self.train_data, self.train_dataloader, self.train_tran_indices, self.train_span_indices, self.train_scls_input_mask = self.get_data_loader(examples=self.train_examples, type='train_data')
        self.eval_data, self.eval_dataloader, self.eval_tran_indices, self.eval_span_indices, self.eval_scls_input_mask = self.get_data_loader(examples=self.eval_examples, type='eval_data')
    
    
    def process_DG(self, DGEDT_train_data, gcls_length):
        aspect_related_paths = [[], [], []]
        for i in range(len(DGEDT_train_data)):
            dgs = []
            dgs.append(torch.tensor(DGEDT_train_data[i]['dependency_graph'][0]))
            dgs.append(torch.tensor(DGEDT_train_data[i]['dependency_graph'][1]))
            dg_ = dgs[0] + dgs[1]
            dg_[dg_>=1] = 1
            dg_ = torch.tensor(dg_)
            dgs.append(dg_)

            for z in range(len(dgs)):
                dg = dgs[z]
                for k in range(len(DGEDT_train_data[i]['span_indices'])):
                    tran_start = DGEDT_train_data[i]['span_indices'][k][0]
                    tran_end = DGEDT_train_data[i]['span_indices'][k][1]

                    input_ids_start = DGEDT_train_data[i]['tran_indices'][tran_start][0]+1
                    input_ids_end = DGEDT_train_data[i]['tran_indices'][tran_end-1][1] +1

                    paths = [[item] for item in range(tran_start, tran_end)]
                    list_ = []
                    for j in range(len(DGEDT_train_data[i]['tran_indices'])):    # 여기서 j는 path의 length를 의미. len(tran_indices[i])는 그냥 넉넉하게 잡은 것.
                        if j == 0 :
                            new_paths = []
                            for item in paths:
                                current_node = item[-1]
                                if len(item) > 1:
                                    last_node = item[-2]
                                else:
                                    last_node = 100000
                                x = (dg[current_node] == 1).nonzero(as_tuple=True)[0]
                                for k in range(x.size(0)):
                                    if x[k] != last_node and x[k]!= current_node:
                                        item_ = item.copy()
                                        item_.append(int(x[k]))
                                        new_paths.append(item_)
                        else:
                            new_paths = []
                            for item in last_new_paths:
                                current_node = item[-1]
                                if len(item) > 1:
                                    last_node = item[-2]
                                else:
                                    last_node = 100000
                                x = (dgs[2][current_node] == 1).nonzero(as_tuple=True)[0]
                                for k in range(x.size(0)):
                                    if x[k] != last_node and x[k]!= current_node:
                                        item_ = item.copy()
                                        item_.append(int(x[k]))
                                        new_paths.append(item_)

                        if len(new_paths) == 0:
                            break

                        last_new_paths = new_paths
                        paths += last_new_paths

                    multiple_path = []
                    aspect_related_paths[z].append(paths)
                    for item in paths:
                        if [item[0], item[-1]] in multiple_path:
                            print('multiple path exists, i: ', i)
                        else:
                            multiple_path += [item[0], item[-1]]
        
        length_L_words = [[], [], []]
        
        A = 2
        for j in range(len(aspect_related_paths)):
            for i in range(len(aspect_related_paths[j])):
                Dict = {}
                for item in aspect_related_paths[j][i]:
                    start_idx, end_idx = DGEDT_train_data[i]['tran_indices'][item[-1]]
                    if len(item)-1 in Dict.keys():
                        Dict[len(item)-1].append([item[-1], start_idx+A, end_idx+A])
                    else:
                        Dict[len(item)-1] = [[item[-1], start_idx+A, end_idx+A]]
                length_L_words[j].append(Dict)

            for i in range(len(DGEDT_train_data)):
                if len(length_L_words[j][i][0]) == 1:
                    assert tokenizer.convert_ids_to_tokens(DGEDT_train_data[i]['aspect_indices'])[1:-1] == \
                    tokenizer.convert_ids_to_tokens(DGEDT_train_data[i]['text_indices'][length_L_words[j][i][0][0][1]-A+1:
                                                                                        length_L_words[j][i][0][0][2]-A+1])
        
        length = gcls_length
            
        gcls_attention_mask = [[],[],[]]    # 2가 GCN용
        for z in range(len(length_L_words)):
            for i in range(len(length_L_words[z])):
                gcls_attention_mask[z].append([[], [], [], []])   # for length 0,1,2,3 each.
                for l in [0,1,2,3]:
                    att_mask = torch.zeros([128])
                    att_mask[0] = 1
                    for j in range(l+1):
                        if j not in length_L_words[z][i].keys():
                            break
                        for item in length_L_words[z][i][j]:
                            att_mask[item[1]:item[2]] = 1
                    gcls_attention_mask[z][i][l] = att_mask

        return gcls_attention_mask
                        
        
    def get_data_loader(self, examples, type='train_data'):
        features = self.convert_examples_to_features(
            examples, self.label_list, self.opt.max_seq_length, self.tokenizer)
        
        if type == 'train_data':
            DGEDT_batches = self.DGEDT_train_batches[0]
            DGEDT_data = self.DGEDT_train_data    # 점검용
        elif type == 'eval_data':
            DGEDT_batches = self.DGEDT_test_batches[0]
            DGEDT_data = self.DGEDT_test_data    # 점검용
        
        batch_size_ = DGEDT_batches['text_indices'].size(0)
        
        all_input_ids_org = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        assert all_input_ids_org.size(0) == batch_size_
        ##############################
        all_input_ids = DGEDT_batches['text_indices']
        ##############################
        
        
        
        all_input_mask_org = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        ##############################
        text_len = torch.sum(DGEDT_batches['text_indices'] != 0, dim=-1)
        all_input_mask = length2mask(text_len, DGEDT_batches['text_indices'].size(1))
        
        # For SCLS
        scls_input_mask = []
        for i in range(batch_size_):
            init_mask = torch.zeros([128, 128])
#             x = (all_input_ids[i] == 102).nonzero(as_tuple=True)[0]
            
#             # 정사각형 attention matrix를 그려보면 코드 이해가 좀 더 쉽다.
#             init_mask[:x[1]+1, :x[1]+1] = torch.ones([x[1]+1, x[1]+1])
            
#             init_mask[x[1]+1:x[2]+1, :x[0]+1] = 1
#             init_mask[x[1]+1:x[2]+1, x[1]+1:x[2]+1] = 1
            
#             init_mask[1] = init_mask[x[1]+1]
            
            scls_input_mask.append(init_mask)
            
        
        ##############################
        
        
        
        all_segment_ids_org = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        ##############################
        all_segment_ids = all_segment_ids_org  # all zeroes of size 128.
        
        ########## When target is appended at the end or the beginning.
        for i in range(batch_size_):
            x = (all_input_ids[i] == 102).nonzero(as_tuple=True)[0]
#             all_segment_ids[i][x[0]+1:x[1]+1] = 1
            
            # For SCLS
            all_segment_ids[i][x[0]+1:x[-1]+1] = 1
            
        ##########
        
        ########## target aspect에만 한 번 segment_ids 1 줘 봄. 
#         for i in range(batch_size_):
#             aspect_start_idx = DGEDT_batches['tran_indices'][i][DGEDT_batches['span_indices'][i][0][0]][0] + 1
#             aspect_end_idx = DGEDT_batches['tran_indices'][i][DGEDT_batches['span_indices'][i][0][1]-1][1] + 1 
#             all_segment_ids[i][aspect_start_idx:aspect_end_idx] = 1
        
        
        
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
        count = 0 
        for i in range(batch_size_):
            if sum(all_input_t_ids_org[i] - all_input_t_ids[i]) != 0:
                count += 1
        assert count/all_input_t_ids.size(0) < 0.02
        # Less than 2% is slightly different between DGEDT and TD-BERT aspect indices. 
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
            input_left_ids[i][aspect_start_idx] = 102    # [SEP]
            
        # 점검
#         for i in range(batch_size_):
#             assert all_input_ids[i][list(input_left_ids[i]).index(102)] == all_input_t_ids[i][1]
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
        
        
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_t_ids,
                             all_input_t_mask, all_segment_t_ids, all_input_without_t_ids, 
                             all_input_without_t_mask, all_segment_without_t_ids, all_input_left_t_ids, all_input_left_t_mask,
                             all_segment_left_t_ids,all_input_right_t_ids, all_input_right_t_mask, all_segment_right_t_ids,
                             input_left_ids, input_left_mask, segment_left_ids, all_input_dg, all_input_dg1, 
                             all_input_dg2, all_input_dg3, all_input_guids)
        
        if type == 'train_data':
            train_data = data
            train_sampler = RandomSampler(data)
            return train_data, DataLoader(train_data, sampler=train_sampler, batch_size=self.opt.train_batch_size), all_tran_indices, all_span_indices, scls_input_mask
        else:
            eval_data = data
            eval_sampler = SequentialSampler(eval_data)
            return eval_data, DataLoader(eval_data, sampler=eval_sampler, batch_size=self.opt.eval_batch_size), all_tran_indices, all_span_indices, scls_input_mask

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
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            if tokens_aspect:  # if not None
                tokens_t = []
                segment_t_ids = []
                tokens_t.append("[CLS]")
                segment_t_ids.append(0)
                for token in tokens_aspect:
                    tokens_t.append(token)
                    segment_t_ids.append(0)
                tokens_t.append("[SEP]")
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
                tokens_without_target.append("[CLS]")
                segment_without_t_ids.append(0)
                for token in tokens_text_without_target:
                    tokens_without_target.append(token)
                    segment_without_t_ids.append(0)
                tokens_without_target.append("[SEP]")
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
                tokens_left_target.append("[CLS]")
                segment_left_t_ids.append(0)
                for token in tokens_text_left_with_target:
                    tokens_left_target.append(token)
                    segment_left_t_ids.append(0)
                tokens_left_target.append("[SEP]")
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
                tokens_right_target.append("[CLS]")
                segment_right_t_ids.append(0)
                for token in tokens_text_right_with_target:
                    tokens_right_target.append(token)
                    segment_right_t_ids.append(0)
                tokens_right_target.append("[SEP]")
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
                tokens_left.append("[CLS]")
                segment_left_ids.append(0)
                for token in tokens_text_left:
                    tokens_left.append(token)
                    segment_left_ids.append(0)
                tokens_left.append("[SEP]")
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
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
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
