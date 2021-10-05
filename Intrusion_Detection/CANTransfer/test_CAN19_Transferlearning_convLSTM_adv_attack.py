import pandas
import numpy as np
import math
import statistics
import keras.backend as K
from time import gmtime, strftime
from keras import applications, Sequential, utils, regularizers
from keras.layers import Dense, BatchNormalization, Dropout, TimeDistributed, LSTM, Flatten, Conv2D, MaxPooling2D, ConvLSTM2D, Conv3D, Activation
from keras.regularizers import L1L2
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint, CSVLogger
from layer_norm import LayerNorm1D
import os
from imblearn.under_sampling import RandomUnderSampler
import glob, os

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0],True)


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


filename = "dataset/combined_data/2nd_labeled_simple_cleaned"
# choose a number of time steps
n_steps = 16
n_features = 11
n_seq = 4
n_substeps = 4
batch_size = 128
num_classes = 2
epochs = 1


def readdata(filename):
    print("Creating npy files...")
    data = pandas.read_csv(filename)
    count_class_0, count_class_1, count_class_2, count_class_3, = data.Label.value_counts()
    data_class_0 = data[data['Label'] == 0]
    data_class_1 = data[data['Label'] == 1]
    data_class_2 = data[data['Label'] == 2]
    data_class_3 = data[data['Label'] == 3]
    data_class_2['Label'] = 1
    data_class_3['Label'] = 1

    ########################class 0############################################
    data_class_0_under = data_class_0.head(count_class_1)
    D0_train, D0_val = train_test_split(data_class_0_under, test_size=0.20, random_state=42, shuffle=False)
    D0_train, D0_test = train_test_split(D0_train, test_size=0.05, random_state=42, shuffle=False)
    print('class 0:', len(D0_train), len(D0_val), len(D0_test))
    ########################class 0############################################

    ########################class 1############################################
    D1_train, D1_val = train_test_split(data_class_1, test_size=0.20, random_state=42, shuffle=False)
    D1_train, D1_test = train_test_split(D1_train, test_size=0.05, random_state=42, shuffle=False)
    print('class 1:', len(D1_train), len(D1_val), len(D1_test))
    ########################class 1############################################

    ########################class 2############################################
    D2_train, D2_val = train_test_split(data_class_2, test_size=0.20, random_state=42, shuffle=False)
    D2_train, D2_test = train_test_split(D2_train, test_size=0.05, random_state=42, shuffle=False)
    print('class 2:', len(D2_train), len(D2_val), len(D2_test))
    test_class_2 = pandas.concat([D0_test, D2_test], axis=0).sort_index()
    X_test2, y_test2 = split_sequences(test_class_2.values, n_steps)
    print('class 2 Sequence:', X_test2.shape, y_test2.shape)
    X_test2 = X_test2.reshape((X_test2.shape[0], n_substeps, 1, n_seq, n_features))
    X_test2 = X_test2.astype('float32')
    print('class 2 Reshape:', X_test2.shape, y_test2.shape)
    y_test2 = to_categorical(y_test2)
    ########################class 2############################################

    ########################class 3############################################
    D3_train, D3_val = train_test_split(data_class_3, test_size=0.20, random_state=42, shuffle=False)
    D3_train, D3_test = train_test_split(D3_train, test_size=0.05, random_state=42, shuffle=False)
    print('class 3:', len(D3_train), len(D3_val), len(D3_test))
    test_class_3 = pandas.concat([D0_test, D3_test], axis=0).sort_index()
    X_test3, y_test3 = split_sequences(test_class_3.values, n_steps)
    print('class 3 Sequence:', X_test3.shape, y_test3.shape)
    X_test3 = X_test3.reshape((X_test3.shape[0], n_substeps, 1, n_seq, n_features))
    X_test3 = X_test3.astype('float32')
    print('class 3 Reshape:', X_test3.shape, y_test3.shape)
    y_test3 = to_categorical(y_test3)
    ########################class 3############################################

    ########################Train Val Test Split###############################
    train_data = pandas.concat([D0_train, D1_train], axis=0).sort_index()
    val_data = pandas.concat([D0_val, D1_val], axis=0).sort_index()
    test_data = pandas.concat([D0_test, D1_test], axis=0).sort_index()
    print('Train Val Test Split:', len(train_data), len(val_data), len(test_data))
    ########################Train Val Test Split###############################

    ########################Split Sequences####################################
    X_train, y_train = split_sequences(train_data.values, n_steps)
    X_val, y_val = split_sequences(val_data.values, n_steps)
    X_test, y_test = split_sequences(test_data.values, n_steps)
    print('Split Sequences:', X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    ########################Split Sequences####################################

    ########################Reshape Sequences##################################

    # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
    X_train = X_train.reshape((X_train.shape[0], n_substeps, 1, n_seq, n_features))
    X_val = X_val.reshape((X_val.shape[0], n_substeps, 1, n_seq, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_substeps, 1, n_seq, n_features))
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    print('Reshape Sequences:', X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    # print(X_train.shape, y_train.shape)
    # print(X_val.shape, y_val.shape)
    # print(X_test.shape, y_test.shape)
    ########################Reshape Sequences##################################

    ########################One Hot Vector#####################################
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)
    ########################One Hot Vector#####################################

    # ########################Class Weight#######################################
    # y_ints = [y.argmax() for y in y_train]
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
    # print(class_weights)
    # ########################Class Weight#######################################

    return X_train, y_train, X_val, y_val, X_test, y_test, X_test2, y_test2, X_test3, y_test3


def convlstm():
    L1L2(l1=0.0001, l2=0.0001)
    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', activation='relu',
                         return_sequences=True, input_shape=(n_seq, 1, n_substeps, n_features),
                         kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(ConvLSTM2D(filters=80, kernel_size=(3, 3), padding='same', activation='relu',
                         return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(ConvLSTM2D(filters=120, kernel_size=(3, 3), padding='same', activation='relu',
                         return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(ConvLSTM2D(filters=160, kernel_size=(3, 3), padding='same', activation='relu',
                         return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(ConvLSTM2D(filters=200, kernel_size=(3, 3), padding='same', activation='relu',
                         return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(ConvLSTM2D(filters=240, kernel_size=(3, 3), padding='same', activation='relu',
                         return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same',
                     data_format='channels_last', kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', padding='same',
                     data_format='channels_last', kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def testmodel(model, X_test, y_test, model_location):
    try:
        model.load_weights(model_location)
        Y_test = np.argmax(y_test, axis=1)
        y_pred = model.predict_classes(X_test)
        print("\n", classification_report(Y_test, y_pred,digits=4))
        # scores = model.evaluate(X_test, y_test, verbose=0)
        # print('Test loss:', scores[0])
        # print('Test accuracy:', scores[1])
    except Exception as e:
        print(e)
import sys
sys.path.append("E:\\Project\\Kari_anomaly_2\\Adversarial_attacks\\cleverhans")
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent


class flags_struct:
    def __init__(self,
                 nb_epochs: int,
                 runs: int,
                 eps: float,
                 adv_train: bool,
                 is_categorical: bool,
                 batch_size: int,
                 UCR_datasets: list):
        self.nb_epochs = nb_epochs
        self.runs = runs
        self.eps = eps
        self.adv_train = adv_train
        self.is_categorical = is_categorical
        self.batch_size = batch_size
        self.UCR_datasets = UCR_datasets


FLAGS = flags_struct(nb_epochs=2000,
                     runs=3,
                     eps=0.1,
                     adv_train=False,
                     is_categorical=False,
                     batch_size=1024,
                     UCR_datasets=['50words'])
def FGSM(trained_model):
    X_train, y_train, X_val, y_val, X_test, y_test, X2, y2, X3, y3 = readdata("dataset/combined_data/2nd_labeled_simple_cleaned.csv")
    X_test = fast_gradient_method(trained_model, X_test, FLAGS.eps, np.inf,y=tf.argmax(tf.cast(y_test, dtype=tf.int32),1))
    X2 = fast_gradient_method(trained_model, X2, FLAGS.eps, np.inf,y=tf.argmax(tf.cast(y2, dtype=tf.int32),1))
    X3 = fast_gradient_method(trained_model, X3, FLAGS.eps, np.inf,y=tf.argmax(tf.cast(y3, dtype=tf.int32),1))
    # X_test = projected_gradient_descent(trained_model, X_test, FLAGS.eps, 0.01, 40, np.inf)
    # X2 = projected_gradient_descent(trained_model, X2, FLAGS.eps, 0.01, 40, np.inf)
    # X3 = projected_gradient_descent(trained_model, X3, FLAGS.eps, 0.01, 40, np.inf)
    for file in glob.glob("Checkpoint/TLConvlstm_oneshot_2019-09-11-05-59-31/5*.h5"):
        print("*******************************************************************")
        model_location = file  # "Checkpoint/TLConvlstm_oneshot_2019-09-12-03-41-35/50.h5"
        print("Testing:", model_location)
        print("\nTesting Class 1...")
        testmodel(trained_model, X_test, y_test, model_location)
        print("\nTesting Class 2...")
        testmodel(trained_model, X2, y2, model_location)
        print("\nTesting Class 3...")
        testmodel(trained_model, X3, y3, model_location)
        print("*******************************************************************")

def PGD(trained_model):
    X_train, y_train, X_val, y_val, X_test, y_test, X2, y2, X3, y3 = readdata("dataset/combined_data/2nd_labeled_simple_cleaned.csv")
    X_test = projected_gradient_descent(trained_model, X_test, FLAGS.eps, 0.01, 40, np.inf, y=tf.argmax(tf.cast(y_test, dtype=tf.int32),1))
    X2 = projected_gradient_descent(trained_model, X2, FLAGS.eps, 0.01, 40, np.inf, y=tf.argmax(tf.cast(y2, dtype=tf.int32),1))
    X3 = projected_gradient_descent(trained_model, X3, FLAGS.eps, 0.01, 40, np.inf, y=tf.argmax(tf.cast(y3, dtype=tf.int32),1))
    for file in glob.glob("Checkpoint/TLConvlstm_oneshot_2019-09-11-05-59-31/5*.h5"):
        print("*******************************************************************")
        model_location = file  # "Checkpoint/TLConvlstm_oneshot_2019-09-12-03-41-35/50.h5"
        print("Testing:", model_location)
        print("\nTesting Class 1...")
        testmodel(trained_model, X_test, y_test, model_location)
        print("\nTesting Class 2...")
        testmodel(trained_model, X2, y2, model_location)
        print("\nTesting Class 3...")
        testmodel(trained_model, X3, y3, model_location)
        print("*******************************************************************")

def NoAttack(trained_model):
    X_train, y_train, X_val, y_val, X_test, y_test, X2, y2, X3, y3 = readdata("dataset/combined_data/2nd_labeled_simple_cleaned.csv")
    for file in glob.glob("Checkpoint/TLConvlstm_oneshot_2019-09-11-05-59-31/5*.h5"):
        print("*******************************************************************")
        model_location = file  # "Checkpoint/TLConvlstm_oneshot_2019-09-12-03-41-35/50.h5"
        print("Testing:", model_location)
        print("\nTesting Class 1...")
        testmodel(trained_model, X_test, y_test, model_location)
        print("\nTesting Class 2...")
        testmodel(trained_model, X2, y2, model_location)
        print("\nTesting Class 3...")
        testmodel(trained_model, X3, y3, model_location)
        print("*******************************************************************")


if __name__ == "__main__":
    trained_model = convlstm()
    NoAttack(trained_model)
    # FGSM(trained_model)
    # PGD(trained_model)

