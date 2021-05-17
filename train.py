import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras import losses
from utils import INPUT_SHAPE, batch_generator, recurrent_batch_generator
import argparse
import os
import model

np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set 
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))
    data_df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    model_dict = {
        "PilotNet": model.build_pilotnet,
        "SmallConvNet": model.build_smallconvnet,
        "RNN": model.build_rnn,
        "LSTM": model.build_lstm
    }

    model_builder = model_dict[args.model]

    return model_builder(args)


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    # Define checkpoint callback to save weights
    checkpoint = ModelCheckpoint(args.model + '-model-epoch-{epoch:03d}-valloss-{val_loss:03f}.h5',
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=args.save_best_only,
                                    mode='auto')
        
    # Try to load checkpoint weights
    if len(args.checkpoint) != 0:
        try:
            model.load_weights(args.checkpoint)
        except:
            print("Invalid checkpoint path. Please try again.")
    print("!!!!!!", X_train.shape)
    # model.summary()
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=args.learning_rate))
    model.summary()
    if args.model in ["RNN", "LSTM"]:
        # Train RNN
        gen = recurrent_batch_generator(args.data_dir, X_train, y_train, args.batch_size, True)
        model.fit(x=gen,
                steps_per_epoch=args.samples_per_epoch,
                epochs=args.nb_epoch,
                max_queue_size=1,
                validation_data=recurrent_batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                validation_steps=len(X_valid),
                callbacks=[checkpoint],
                verbose=1)
        
    else:
        # Train CNN
        model.fit(x=batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                steps_per_epoch=args.samples_per_epoch,
                epochs=args.nb_epoch,
                max_queue_size=1,
                validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                validation_steps=len(X_valid),
                callbacks=[checkpoint],
                verbose=1)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='..\data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=64)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    parser.add_argument('-m', help='model architecture',    dest='model',             type=str,   default="PilotNet", choices=["PilotNet", "SmallConvNet", "RNN", "LSTM"])
    parser.add_argument('-c', help='load checkpoint',       dest='checkpoint',        type=str,   default="")
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()

