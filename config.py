'''
File path
'''
MODEL_PATH='./model/'
LOG_PATH='./log/'

'''
model list
'''
# RNN_BASIC='RNN'
RNN_LSTM='RNN_LSTM'
RNN_GRU='RNN_GRU'
# BD_RNN_BASIC='BRNN_BASIC'
BD_LSTM='BRNN_LSTM'
BD_GRU="BRNN_GRU"

rnn_model_list=[RNN_LSTM,RNN_GRU,BD_LSTM,BD_GRU]

'''
dataset list
'''
IMDB_DATASET="IMDB"
ROTTEN_TOMATO_DATASET="ROTTEN_TOMATO"
# dataset_list=[IMDB_DATASET,ROTTEN_TOMATO_DATASET]
dataset_list=[IMDB_DATASET]

'''
configuration
'''
IMDB_NUM_WORDS=100000

IMDB_SEQ_LEN=100



def pprint_args(FLAGS):
    print("\nParameters:")
    for attr, value in sorted(vars(FLAGS).items()):
        print("{}={}".format(attr.upper(), value))
    print("")



# def getModel(modelName):

