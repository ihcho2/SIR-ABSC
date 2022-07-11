import logging
import torch
import os

import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch import nn

logger = logging.getLogger(__name__)

def load_model_ER_1(model, checkpoint, mode='student', verbose=True, DEBUG=False):
    """

    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param train_mode:
    :param verbose:
    :return:
    """

    local_rank = -1
    if checkpoint in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (model._get_name(), checkpoint))
        model_state_dict = torch.load(checkpoint)
#         old_keys = []
#         new_keys = []
#         for key in model_state_dict.keys():
#             new_key = None
#             if 'gamma' in key:
#                 new_key = key.replace('gamma', 'weight')
#             if 'beta' in key:
#                 new_key = key.replace('beta', 'bias')
#             if key.startswith('module.'):
#                 new_key = key.replace('module.', '')
#             if new_key:
#                 old_keys.append(key)
#                 new_keys.append(new_key)
#         for old_key, new_key in zip(old_keys, new_keys):
#             model_state_dict[new_key] = model_state_dict.pop(old_key)
            
#         model_state_dict_org = model_state_dict.copy()

        del_keys = []
        keep_keys = []
        
        
        model_keys = model.state_dict().keys()
        
#         if student_layer_initialization != None:
#             assert args.student_hidden_layers == len(student_layer_initialization)

#             for keys in model_keys:
#                 if 'scc' in keys:
#                     is_Theseus = True

#             if is_Theseus == False:
#                 for i in range(len(student_layer_initialization)):
#                     for keys in model_keys:
#                         if f'bert.encoder.layer.{i}.' in keys:
#                             model_state_dict[keys] = model_state_dict_org[keys.replace(str(i), 
#                                                                                       str(student_layer_initialization[i]-1))]
#             else:
#                 for i in range(len(student_layer_initialization)):
#                     for keys in model_keys:
#                         if f'bert.encoder.scc_layer.{i}.' in keys:
#                             model_state_dict[keys] = model_state_dict_org[keys.replace(f'scc_layer.{i}', 
#                                                                               f'layer.{student_layer_initialization[i]-1}')]

#         for t in list(model_state_dict.keys()):
#             if t not in model_keys:
#                 del model_state_dict[t]
#                 del_keys.append(t)
#             else:
#                 keep_keys.append(t)
        
        model.load_state_dict(model_state_dict)
        
#         if student_layer_initialization != None:
#             if is_Theseus == False:
#                 for keys in model_state_dict.keys():
#                     for i in range(len(student_layer_initialization)):
#                         if '.'+str(i)+'.' in keys:
#                             assert model_state_dict[keys].mean() == model_state_dict_org[keys.replace(str(i),                                                                                           str(student_layer_initialization[i]-1))].mean()
#             else:
#                 for keys in model_state_dict.keys():
#                     for i in range(len(student_layer_initialization)):
#                         if '.layer.'+str(i)+'.' in keys:
#                             assert model_state_dict[keys].mean() == model_state_dict_org[keys].mean()
#                         if '.scc_layer.'+str(i)+'.' in keys:
#                             assert model_state_dict[keys].mean() == model_state_dict_org[keys.replace(f'scc_layer.{i}',
#                                                                             f'layer.{student_layer_initialization[i]-1}')].mean()
                    
#         logger.info('delete %d layers, keep %d layers' % (len(del_keys), len(keep_keys)))
        
    return model

def load_model_roberta_rpt(model, checkpoint1, checkpoint2, checkpoint3, checkpoint4=None, mode='student', verbose=True, DEBUG=False):
    """

    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param train_mode:
    :param verbose:
    :return:
    """

    local_rank = -1
    if checkpoint1 in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint1):
            raise ValueError('checkpoint %s not exist' % checkpoint1)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (model._get_name(), checkpoint1))
        model_state_dict_1 = torch.load(checkpoint1)
        model_state_dict_2 = torch.load(checkpoint2)
        model_state_dict_3 = torch.load(checkpoint3)
        if checkpoint4 != None:
            model_state_dict_4 = torch.load(checkpoint4)
        
#         model_state_dict_1_ = {}
#         model_state_dict_2_ = {}
#         model_state_dict_3_ = {}
        
#         for key in model_state_dict_1.keys():
#             new_key = 'bert.'+key
#             model_state_dict_1_[new_key] = model_state_dict_1[key]
            
#         for key in model_state_dict_2.keys():
#             new_key = 'bert.'+key
#             model_state_dict_2_[new_key] = model_state_dict_2[key]
            
#         for key in model_state_dict_3.keys():
#             new_key = 'bert.'+key
#             model_state_dict_3_[new_key] = model_state_dict_3[key]

        model_keys = model.state_dict().keys()
        
        merged_model_state_dict = model_state_dict_1.copy()
        
        
        
#         for key in model_keys:
#             for i in range(12,24):
#                 if '.'+str(i)+'.' in key:
#                     merged_model_state_dict[key] = model_state_dict_2[key.replace(f'bert.encoder.layer.{i}.',
#                                                                                   f'encoder.layer.{i-12}.')]
#             for i in range(24,36):
#                 if '.'+str(i)+'.' in key:
#                     merged_model_state_dict[key] = model_state_dict_3[key.replace(f'bert.encoder.layer.{i}.',
#                                                                                   f'encoder.layer.{i-24}.')]
                    
        for key in model_keys:
            for i in range(12,24):
                if '.'+str(i)+'.' in key:
                    merged_model_state_dict[key] = model_state_dict_2[key.replace(f'.{i}.', f'.{i-12}.')]
            for i in range(24,36):
                if '.'+str(i)+'.' in key:
                    merged_model_state_dict[key] = model_state_dict_3[key.replace(f'.{i}.', f'.{i-24}.')]
            if checkpoint4 != None:
                for i in range(36,48):
                    if '.'+str(i)+'.' in key:
                        merged_model_state_dict[key] = model_state_dict_4[key.replace(f'.{i}.', f'.{i-36}.')]
                
        model.load_state_dict(merged_model_state_dict)
        
    return model

def load_model_MoE(model, checkpoint1, checkpoint2, checkpoint3, checkpoint4=None, checkpoint5 =None, mode='student', verbose=True, DEBUG=False):
    """

    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param train_mode:
    :param verbose:
    :return:
    """

    local_rank = -1
    if checkpoint1 in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint1):
            raise ValueError('checkpoint %s not exist' % checkpoint1)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (model._get_name(), checkpoint1))
        model_state_dict_1 = torch.load(checkpoint1)
        model_state_dict_2 = torch.load(checkpoint2)
        model_state_dict_3 = torch.load(checkpoint3)
        if checkpoint4 != None:
            model_state_dict_4 = torch.load(checkpoint4)
        if checkpoint5 != None:
            model_state_dict_5 = torch.load(checkpoint5)
            
            
        
#         model_state_dict_1_ = {}
#         model_state_dict_2_ = {}
#         model_state_dict_3_ = {}
        
#         for key in model_state_dict_1.keys():
#             new_key = 'bert.'+key
#             model_state_dict_1_[new_key] = model_state_dict_1[key]
            
#         for key in model_state_dict_2.keys():
#             new_key = 'bert.'+key
#             model_state_dict_2_[new_key] = model_state_dict_2[key]
            
#         for key in model_state_dict_3.keys():
#             new_key = 'bert.'+key
#             model_state_dict_3_[new_key] = model_state_dict_3[key]

        model_keys = model.bert.state_dict().keys()
        
        merged_model_state_dict = model_state_dict_1.copy()
        
        
#         for key in model_keys:
#             for i in range(12,24):
#                 if '.'+str(i)+'.' in key:
#                     merged_model_state_dict[key] = model_state_dict_2[key.replace(f'bert.encoder.layer.{i}.',
#                                                                                   f'encoder.layer.{i-12}.')]
#             for i in range(24,36):
#                 if '.'+str(i)+'.' in key:
#                     merged_model_state_dict[key] = model_state_dict_3[key.replace(f'bert.encoder.layer.{i}.',
#                                                                                   f'encoder.layer.{i-24}.')]
                    
        for key in model_keys:
            for i in range(12,24):
                if '.'+str(i)+'.' in key:
                    merged_model_state_dict[key] = model_state_dict_2[key.replace(f'.{i}.', f'.{i-12}.')]
            for i in range(24,36):
                if '.'+str(i)+'.' in key:
                    merged_model_state_dict[key] = model_state_dict_3[key.replace(f'.{i}.', f'.{i-24}.')]
            if checkpoint4 != None:
                for i in range(36,48):
                    if '.'+str(i)+'.' in key:
                        merged_model_state_dict[key] = model_state_dict_4[key.replace(f'.{i}.', f'.{i-36}.')]
                        
        if checkpoint5 != None:
            for key in model_keys:
                for i in range(0,4):
                    if '.'+str(i)+'.' in key:
                        merged_model_state_dict[key] = model_state_dict_5['bert.'+key.replace(f'.{i}.', f'.{i}.')]
                for i in range(16,20):
                    if '.'+str(i)+'.' in key:
                        merged_model_state_dict[key] = model_state_dict_5['bert.'+key.replace(f'.{i}.', f'.{i-12}.')]
                for i in range(32,36):
                    if '.'+str(i)+'.' in key:
                        merged_model_state_dict[key] = model_state_dict_5['bert.'+key.replace(f'.{i}.', f'.{i-24}.')]
                
        
        #이거 고쳐야 할 듯. model.load_state_dict 로 ㄱ ㄱ
        model.bert.load_state_dict(merged_model_state_dict)
#         model.load_state_dict(merged_model_state_dict)
        
        
    return model

def load_model_init(model, checkpoint, args, mode='exact', train_mode='finetune', verbose=True, DEBUG=False):
    """

    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param train_mode:
    :param verbose:
    :return:
    """
    
    checkpoint_2 = './init_encoder.pkl'
    n_gpu = args.n_gpu
    device = args.device
    local_rank = -1
    if checkpoint in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (model._get_name(), checkpoint))
        model_state_dict_2 = torch.load(checkpoint)
        model_state_dict = torch.load(checkpoint_2)
        
        for key in model_state_dict_2:
            if key in model_state_dict.keys():
                model_state_dict[key] = model_state_dict_2[key]
        old_keys = []
        new_keys = []
        for key in model_state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            model_state_dict[new_key] = model_state_dict.pop(old_key)

        del_keys = []
        keep_keys = []
        if mode == 'exact':
            pass
        elif mode == 'encoder':
            for t in list(model_state_dict.keys()):
                if 'classifier' in t or 'cls' in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'classifier':
            for t in list(model_state_dict.keys()):
                if 'classifier' not in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'student':
            model_keys = model.state_dict().keys()
            for t in list(model_state_dict.keys()):
                if t not in model_keys:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        else:
            raise ValueError('%s not available for now' % mode)
        model.load_state_dict(model_state_dict)
        if mode != 'exact':
            logger.info('delete %d layers, keep %d layers' % (len(del_keys), len(keep_keys)))
        if DEBUG:
            print('deleted keys =\n {}'.format('\n'.join(del_keys)))
            print('*' * 77)
            print('kept keys =\n {}'.format('\n'.join(keep_keys)))

    if args.fp16:
        logger.info('fp16 activated, now call model.half()')
        model.half()
    model.to(device)

    if train_mode != 'finetune':
        if verbose:
            logger.info('freeze BERT layer in DEBUG mode')
        model.set_mode(train_mode)

    if local_rank != -1:
        raise NotImplementedError('not implemented for local_rank != 1')
    elif n_gpu > 1:
        logger.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    return model
