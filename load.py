'''
loads input dataset
'''

from keras.datasets import imdb
from keras.preprocessing import sequence

'''
load imdb training, testing data 
return length of the sequence , (training data, label), (testing data, label)
'''
def load_imdb_input(num_words,seq_len):

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    # "Pads each sequence to the same length (length of the longest sequence)
    x_train = sequence.pad_sequences(x_train, maxlen=seq_len)
    x_test = sequence.pad_sequences(x_test, maxlen=seq_len)

    return (x_train,y_train),(x_test,y_test)



