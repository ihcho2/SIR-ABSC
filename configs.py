# Time: 2019-3-7 20:15:18
# Author: gaozhengjie

import argparse
from datetime import datetime

def get_config():
    parser = argparse.ArgumentParser()
    TIMESTAMP = "{0:%Y-%m-%d--%H-%M-%S/}".format(datetime.now())

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default='log/'+TIMESTAMP,
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model_save_path",
                        default='save_model/' + TIMESTAMP,
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
    parser.add_argument("--do_train",
                        default=True,
                        help="Whether to run training.")
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
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
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
                        default=[],
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
    parser.add_argument('--gcn_data_dir', type=str, default='dataset/Restaurants')
    parser.add_argument('--vocab_dir', type=str, default='./datasets/semeval14/laptops/3way')
    parser.add_argument('--glove_dir', type=str, default='./datasets/glove')
    parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--gcn_hidden_dim', type=int, default=50, help='GCN mem dim.')
    parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
    parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class used in gcn_only (CDT).')
    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--direct', default=False)
    parser.add_argument('--loop', default=True)
    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adamax', help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')

    return parser.parse_args()
