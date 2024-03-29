# Time: 2022 - 7 - 4
# Author: Ikhyun Cho and Yoonhwa Jung

import argparse
from datetime import datetime

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_config_2():
    parser = argparse.ArgumentParser()
    TIMESTAMP = "{0:%Y-%m-%d--%H-%M-%S/}".format(datetime.now())

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--input_format",
                        default=None,
                        type=str,
                        required=True,
                        help="The input format of the data.")
    parser.add_argument("--bert_config_file",
                        default=None, # /home/ikhyuncho23/GoBERTa/uncased_L-12_H-768_A-12/bert_config.json
                        type=str,
                        required=False,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--roberta_config_file",
                        default='/home/ikhyuncho23/GoBERTa/roberta_files/config.json',
                        type=str,
                        required=False,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default='/home/ikhyuncho23/GoBERTa/roberta_files/vocab.json', # For BERT: /home/ikhyuncho23/GoBERTa/uncased_L-12_H-768_A-12/vocab.txt
                        type=str,
                        required=False,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default='log/'+TIMESTAMP,
                        type=str,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--eval_start_epoch",
                        default=0,
                        type=int,
                        help="When to start evaluation. Used for saving training time.")
    parser.add_argument("--do_train",
                        default=True,
                        help="Whether to run training.")
    parser.add_argument('--save_init_model',
                        type= boolean_string, default = False)
    parser.add_argument("--do_eval",
                        default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        default=False,
                        help="Whether to run eval on the test set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--VDC_threshold",
                        default=0.8,
                        type=float,
                        help="The VDC threshold.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--K",
                        default=10,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--gpu_id',
                        default=[0],
                        type=list,
                        help=u'Enter the GPU number to specify')

    # parser.add_argument('--model_name', default='lstm', type=str)
    parser.add_argument('--model_name', default='fc', type=str)  # Full connection model, that is, the output of bert is followed by full connection
    parser.add_argument('--embed_dim', default=768, type=int)
    parser.add_argument('--n_filters', default=100, type=int)
    parser.add_argument('--filter_sizes', default=[1, 2, 3, 4], type=list)
    parser.add_argument('--dropout', default=0, type=float)
    # parser.add_argument('--output_dim', default=3, type=int)  # CNN里面的输出维度，也就是标签的个数
    parser.add_argument('--hidden_dim', default=300, type=int)  # It used to be 150, and it was changed to 300 by running aen_bert
    parser.add_argument('--lstm_layers', default=1, type=int)
    parser.add_argument('--lstm_mean', default='maxpool', type=str)
    parser.add_argument('--keep_dropout', default=0.1, type=float)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--para_LSR', default=0.2, type=float)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    
    ###### By Cho & Jung
    parser.add_argument("--parser_info",
                        type=str,
                        help="surface distance or dependency graph distance"),
    
    parser.add_argument('--pb', type=int, default=1, help='Not used in final model.')
    
    parser.add_argument('--use_hVDC',
                        type= boolean_string, default = False)
    
    parser.add_argument('--use_DEP',
                        type= boolean_string, default = False)
    
    parser.add_argument('--use_POS',
                        type= boolean_string, default = False)
    
    parser.add_argument('--do_auto',
                        type= boolean_string, default = False)
    
    parser.add_argument('--constant_vdc',
                        type=lambda s: [int(item) for item in s.split(',')],
                        default = None,
                        help='VDC configuration for each layer.')
    
    parser.add_argument('--auto_layers',
                        type=lambda s: [int(item) for item in s.split(',')],
                        default = None,
                        help='VDC configuration for each layer.')
    
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature.')
    
    parser.add_argument('--auto_type', type=str, default=None, help='auto_type')
    
    parser.add_argument('--g_config',
                        type=lambda s: [int(item) for item in s.split(',')],
                        default = None,
                        help='g_config.')
    
    parser.add_argument('--embed_dense',
                        type= boolean_string, default = False)
    
    parser.add_argument("--graph_type",
                        type=str,
                        help="surface distance or dependency graph distance"),
    
    parser.add_argument("--g_pooler",
                        type=str,
                        default = 'att',
                        help="The pooler type, one of att, avg, max."),
    
    parser.add_argument("--model_save_path", default='/home/ikhyuncho23/GoBERTa/gcls_auto' ,type=str,
                        help="The output directory where the model checkpoints will be written.")
    
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    
    parser.add_argument('--do_save', type = boolean_string, default=False)
    
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adamax', help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--log_step', type=int, default=5, help='Print log every k steps.')
    parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')

    return parser.parse_args()
