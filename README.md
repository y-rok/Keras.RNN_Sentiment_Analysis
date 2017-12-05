# Keras.RNN_Sentiment_Analysis
RNN and bidirectional RNN training with IMDB movie review data to analyze sentiment

## Requirements

- python3.5
- tensorflow 1.3
- keras 2.1.1

## Usage

### Training model

- If dataset is not in the root folder, download dataset before training.
- 0.1% of the training dataset is used to validate the model. 
- The model is saved in ./model/[modelname_layer]/ whenever the validation accuracy is improved.

```
python3 train.py --h

usage: train.py [-h] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
                --model_name MODEL_NAME --dataset DATASET [--layer LAYER]

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        default: 20
  --batch_size BATCH_SIZE
                        default: 32
  --model_name MODEL_NAME
                        BD -> Bidirectional RNN
                        LIST = RNN_LSTM / RNN_GRU / BRNN_LSTM / BRNN_GRU
  --dataset DATASET, -D DATASET
                        LIST = IMDB 
  --layer LAYER         the structure of neural network
                        ex) 200,100 layer 1 =200, layer 2 = 100
                        (default : 300)
```

Example : 
```
python3 train.py --dataset=IMDB --model_name=BRNN_LSTM --num_epoch=10 --batch_size=32 --layer=50,50
```
### Monitoring through tensorboard

```
tensorboard --logdir=./log/
```

### Evaluating model

- evaluate the saved model with testing dataset

```
python3 eval.py --h

usage: eval.py [-h] --model_path MODEL_PATH --dataset DATASET

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        saved model path
  --dataset DATASET, -D DATASET
                        LIST = IMDB 
```

Example : 
```
python3 eval.py --model_path=model/BRNN_GRU_[300]/02_0.84.hdf5 --dataset=IMDB
```


## References

- https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow


