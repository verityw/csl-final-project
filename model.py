import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, TimeDistributed, RNN, SimpleRNN, LSTM
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

np.random.seed(0)

def build_pilotnet(args):
    """
    Modified NVIDIA model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 2, activation='elu'))
    model.add(Conv2D(36, 5, 2, activation='elu'))
    model.add(Conv2D(48, 5, 2, activation='elu'))
    model.add(Conv2D(64, 3, 1, activation='elu'))
    model.add(Conv2D(64, 3, 1, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def build_smallconvnet(args):
    """
    Smaller convolutional network
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(16, 5, 2, activation='elu'))
    model.add(Conv2D(32, 5, 2, activation='elu'))
    model.add(Conv2D(48, 3, 1, activation='elu'))
    model.add(Conv2D(64, 3, 1, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model

def build_lstm(args):
    "Build long short-term memory RNN"
    model = Sequential()
    model.add(TimeDistributed(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE)))
    model.add(Conv2D(16, 5, 2, activation='elu'))
    model.add(Conv2D(32, 3, 2, activation='elu'))
    model.add(Conv2D(64, 3, 1, activation='elu'))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dropout(args.keep_prob)))
    model.add(TimeDistributed(Dense(units=32,   activation='relu')))
    model.add(LSTM(8))
    model.add(TimeDistributed(Dense(1)))
    model.summary()

    return model

def build_rnn(args):
    "Build RNN (fully connected recurrency)"
    model = Sequential()
    model.add(TimeDistributed(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE)))
    model.add(Conv2D(16, 5, 2, activation='elu'))
    model.add(Conv2D(32, 3, 2, activation='elu'))
    model.add(Conv2D(64, 3, 1, activation='elu'))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dropout(args.keep_prob)))
    model.add(TimeDistributed(Dense(units=32,   activation='relu')))
    model.add(SimpleRNN(8))
    model.add(TimeDistributed(Dense(1)))
    model.summary()

    return model

