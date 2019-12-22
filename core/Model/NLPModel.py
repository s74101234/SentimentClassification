import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, SimpleRNN, GRU, Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.layers import Dropout, Activation, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

def buildRNNModel(max_fatures, input_length, embed_dim, embedding_matrix, num_classes, num_GPU):
    model = Sequential()

    if(embedding_matrix == None):
        model.add(Embedding(max_fatures, embed_dim, input_length = input_length))
    else:
        model.add(Embedding(max_fatures, embed_dim, weights = [embedding_matrix], input_length = input_length, trainable = False))
        
    model.add(SimpleRNN(128))
    model.add(Dense(num_classes, activation = 'softmax'))

    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = 'categorical_crossentropy',
            optimizer = 'rmsprop',
            metrics = ['accuracy'])
    return model

def buildLSTMModel(max_fatures, input_length, embed_dim, embedding_matrix, num_classes, num_GPU):
    model = Sequential()

    if(embedding_matrix == None):
        model.add(Embedding(max_fatures, embed_dim, input_length = input_length))
    else:
        model.add(Embedding(max_fatures, embed_dim, weights = [embedding_matrix], input_length = input_length, trainable = False))
    # DCNN 0.77~0.8
    # model.add(Dropout(0.4))
    # model.add(Conv1D(256, 5, padding = 'valid', activation = 'relu', strides = 1, kernel_regularizer = regularizers.l2(0.01)))
    # model.add(MaxPooling1D(pool_size = 2))
    # model.add(LSTM(128, activation ='tanh', kernel_regularizer = regularizers.l2(0.01)))
    # model.add(Dropout(0.4))

    # # https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py 0.73
    # model.add(Conv1D(128, 5, activation='relu'))
    # model.add(MaxPooling1D(pool_size = 5))
    # model.add(Conv1D(128, 5, activation='relu'))
    # model.add(GlobalMaxPooling1D())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.4))

    model.add(LSTM(128, activation ='tanh'))

    model.add(Dense(num_classes, activation = 'softmax'))

    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = 'categorical_crossentropy',
            optimizer = 'Adam',
            metrics = ['accuracy'])
    return model

def buildGRUModel(max_fatures, input_length, embed_dim, embedding_matrix, num_classes, num_GPU):
    model = Sequential()

    if(embedding_matrix == None):
        model.add(Embedding(max_fatures, embed_dim, input_length = input_length))
    else:
        model.add(Embedding(max_fatures, embed_dim, weights = [embedding_matrix], input_length = input_length, trainable = False))
        
    model.add(GRU(256, return_sequences = True))
    model.add(SimpleRNN(128))
    model.add(Dense(num_classes, activation = 'softmax'))

    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr = 0.001),
            metrics = ['accuracy'])
    return model
