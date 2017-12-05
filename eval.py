import textwrap
import config
import load

from rnn import *
from argparse import ArgumentParser

def build_parser():
    parser = ArgumentParser()
    models_str = ' / '.join(config.rnn_model_list)
    dataset_str=' / '.join(config.dataset_list)

    # parser.add_argument('--model_name', help=textwrap.dedent('''\
    # BD -> Bidirectional RNN
    # '''+'LIST = '+models_str), required=True)

    parser.add_argument('--model_path', help='saved model path', required=True)
    parser.add_argument('--dataset', '-D', help='LIST = '+dataset_str, required=True)


    return parser

def eval(num_words,seq_len,model_path):

    print("load test data")
    _, (test_x, test_y) = load.load_imdb_input(num_words, seq_len)

    rnn=RNN()

    print("load model")
    rnn.load(model_path)

    print("evaluate model")
    rnn.eval(test_x,test_y)

'''
    load and evaluate model
'''
if __name__ == "__main__":

    parser = build_parser()
    FLAGS = parser.parse_args()
    # FLAGS.model_path = FLAGS.model_path.upper()
    FLAGS.dataset=FLAGS.dataset.upper()

    if FLAGS.dataset==config.IMDB_DATASET:
        num_words=config.IMDB_NUM_WORDS
        seq_len=config.IMDB_SEQ_LEN

    eval(num_words,seq_len,FLAGS.model_path)

