import argparse
import textwrap

from rnn import *
import load
import config
import tensorflow as tf
from keras import backend as K


def build_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--num_epochs', default=20, help='default: 20', type=int)
    parser.add_argument('--batch_size', default=128, help='default: 32', type=int)
    models_str = ' / '.join(config.rnn_model_list)
    dataset_str=' / '.join(config.dataset_list)
    parser.add_argument('--model_name', help=textwrap.dedent('''\
    BD -> Bidirectional RNN
    '''+'LIST = '+models_str), required=True)
    parser.add_argument('--dataset', '-D', help='LIST = '+dataset_str, required=True)

    # parser.add_argument('--layer',help=textwrap.dedent('''\
    #     the structure of neural network
    #     ex) 200,100 layer 1 =200, layer 2 = 100
    #     (default : 300) '''), default='300')
    #
    # parser.add_argument('--gpu_id', default=-0, help=textwrap.dedent('''\
    #     the id of gpu used for training
    #     (default : 0) '''), type=int)

    parser.add_argument('--layer', help=textwrap.dedent('''\
         the structure of neural network 
         ex) 200,100 layer 1 =200, layer 2 = 100
         (default : 300) '''), default='300')

    # parser.add_argument('--num_words', help=textwrap.dedent('''\
    #      The number of word to keep in the vocabulary.
    #      default : 10000'''), default=100000, type=int)
    #
    # parser.add_argument('--seq_len', help=textwrap.dedent('''\
    #      The max sequence length we use in the train/valid set
    #      default : 100'''), default=100, type=int)

    return parser


def train(dataset,model_name,layer,num_epochs,batch_size,num_words,seq_len):

    model =None

    # dynamically allocate GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    K.set_session(sess)


    if dataset == config.dataset_list[0]:
        print("Loading data...")
        (train_x, train_y), _ = load.load_imdb_input(num_words, seq_len)

        print("Build model...")
        rnn = RNN(layer, model_name)
        # build model
        rnn.build_imdb_model(num_words)

    print('Train...')
    rnn.train(train_x,train_y,num_epochs,batch_size)


'''
    Train model and save improved models
'''
if __name__ == "__main__":
    parser = build_parser()
    FLAGS = parser.parse_args()
    FLAGS.model_name=FLAGS.model_name.upper()
    FLAGS.dataset=FLAGS.dataset.upper()
    # convert string list to int list
    FLAGS.layer=[int(x) for x in FLAGS.layer.split(',')]

    # print parameter
    config.pprint_args(FLAGS)

    if FLAGS.dataset==config.IMDB_DATASET:
        num_words=config.IMDB_NUM_WORDS
        seq_len=config.IMDB_SEQ_LEN

    rnn=train(FLAGS.dataset, FLAGS.model_name,FLAGS.layer,FLAGS.num_epochs, FLAGS.batch_size,num_words,seq_len)
