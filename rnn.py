import config
from keras.layers import Dense, Embedding,Dropout
from keras.layers import LSTM, GRU,LSTMCell,GRUCell,Bidirectional
from keras.layers import RNN as keras_RNN
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
import os



class RNN(object):
    def __init__(self,layer = None,model_name = None):
            self.layer=layer
            self.model_name=model_name

    def _get_model(self,unit_num,return_seq=True):


        if self.model_name==config.RNN_LSTM:
            return LSTM(unit_num,return_sequences=return_seq,dropout=0.2,recurrent_dropout=0.2)
        elif self.model_name==config.RNN_GRU:
            return GRU(unit_num,return_sequences=return_seq,dropout=0.2,recurrent_dropout=0.2)
        # bidirectional output will be concatenated
        # -> [batch_size, unit_num*2] (2 hidden states from forward and backward networks are concatenated)
        elif self.model_name==config.BD_LSTM:
            return Bidirectional(LSTM(unit_num,return_sequences=return_seq,dropout=0.2,recurrent_dropout=0.2))
        elif self.model_name==config.BD_GRU:
            return Bidirectional(GRU(unit_num,return_sequences=return_seq,dropout=0.2,recurrent_dropout=0.2))

        # cells=[]
        # if self.model_name == config.RNN_LSTM or self.model_name == config.BD_LSTM:
        #     for unit_num in self.layer:
        #         cells.append(LSTMCell(unit_num))
        #     if self.model_name==config.RNN_LSTM:
        #         return keras_RNN(cells)
        #     elif self.model_name==config.BD_LSTM:
        #         return Bidirectional()
        #
        # elif self.model_name == config.RNN_GRU or self.model_name == config.BD_GRU:
        #     for unit_num in self.layer:
        #         cells.append(GRUCell(unit_num))
        #         return keras_RNN(cells)

    def _get_log_path(self):
        return config.LOG_PATH+self.model_name+'_'+str(self.layer).replace(" ","")+"/"

    def _get_model_path(self):
        return config.MODEL_PATH+self.model_name+'_'+str(self.layer).replace(" ","")+"/"

    def build_imdb_model(self,num_words):
        self.model = Sequential()
        # [batch_size, seq_len] -> [batch_size, seq_len, output_dim(self.layer[0])]
        self.model.add(Embedding(num_words, self.layer[0]))
        # self.model.add(self._get_model())

        for index, unit_num in enumerate(self.layer):
            if index!=len(self.layer)-1:
                # return hidden states of all sequences [batch_size, seq_len, unit_num]
                self.model.add(self._get_model(unit_num,return_seq=True))
            else:
                # return the final hidden state [batch_size, unit_num]
                self.model.add(self._get_model(unit_num,return_seq=False))

        # output layer
        # [batch_size, unit_num] -> [batch_size,1]
        self.model.add(Dense(1, activation='sigmoid'))


        # if self.model_name == config.RNN_GRU or self.model_name == config.RNN_LSTM:
        #     # output layer
        #     # [batch_size, unit_num] -> [batch_size,1]
        #     self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        print(self.model.summary())



    '''
        While training model, check if the validation accuracy of model improves and only save improved model
    '''
    def train(self,train_x,train_y,num_epochs,batch_size):

        # create folder where the model will be saved
        if not os.path.exists(self._get_model_path()):
            os.makedirs(self._get_model_path())
        if not os.path.exists(self._get_log_path()):
            os.makedirs(self._get_log_path())

        # model save description - https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
        checkpointer = ModelCheckpoint(filepath=self._get_model_path()+'{epoch:02d}_{val_acc:.2f}.hdf5', verbose=1,monitor='val_acc', save_best_only=True)
        tensorboard = TensorBoard(log_dir=self._get_log_path())
        # 10% training dataset is used for validation dataset
        self.model.fit(train_x, train_y,
                              batch_size=batch_size,
                              epochs=num_epochs,
                              validation_split=0.1,
                              callbacks=[checkpointer,tensorboard])

    def load(self,model_path):
        self.model=load_model(model_path)


    def eval(self,test_x,test_y):

        score, acc = self.model.evaluate(test_x, test_y)
        print('Test score:', score)
        print('Test accuracy:', acc)














