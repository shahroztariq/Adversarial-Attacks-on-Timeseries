import pandas
import numpy as np
import math
import statistics
import keras.backend as K
from time import gmtime, strftime
from keras import applications, Sequential, regularizers

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



physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0],True)

def split_sequences(sequences,n_steps):
    X,y = list(),list()
    for i in range (len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix,:-1],sequences[end_ix-1,-1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

filename="dataset/combined_data/2nd_labeled_simple_cleaned"

# choose a number of time steps
n_steps = 16
n_features = 11
n_seq = 4
n_substeps = 4
batch_size=128
num_classes=2
epochs=50

#
# # split into samples
# # data=pandas.read_csv("dataset/combined_data/test.csv")
# if os.path.exists(filename+"_.npy") and os.path.isfile(filename+"_X.npy") \
#         and os.path.exists(filename+"_y.npy") and os.path.isfile(filename+"_y.npy"):
#     print("loading npy files...")
#     X=np.load(filename + '_X.npy')
#     y=np.load(filename + '_y.npy')
#     print(X.shape, y.shape)
#     X = X.reshape((X.shape[0], n_substeps, 1, n_seq, n_features))
#     print(X.shape, y.shape)
#
# else:
#     print("Creating npy files...")
#     data=pandas.read_csv("dataset/combined_data/2nd_labeled_simple_cleaned.csv")
#     count_class_0, count_class_1,count_class_2, count_class_3, = data.Label.value_counts()
#     minimum=min(count_class_0,count_class_1,count_class_2,count_class_3)
#     data_class_0 = data[data['Label'] == 0]
#     data_class_1 = data[data['Label'] == 1]
#     data_class_2 = data[data['Label'] == 2]
#     data_class_3 = data[data['Label'] == 3]
#     data_class_2['Label'] = 1
#     data_class_3['Label'] = 1
#     # print(data_class_1.head(5))
#     # print(data_class_2.head(5))
#     # print(data_class_3.head(5))
#     # exit(0)
#     data_class_0_under = data_class_0.sample(count_class_1)
#     # data_class_1_under = data_class_1.sample(count_class_1)
#     # data_class_2_under = data_class_2.sample(count_class_1)
#     # data_class_3_under = data_class_3.sample(count_class_1)
#     data_undersampled = pandas.concat([data_class_0_under, data_class_1], axis=0)
#     # data_undersampled = pandas.concat([data_class_0_under, data_class_1,data_class_2,data_class_3], axis=0)
#     # data_undersampled = pandas.concat([data_class_0_under, data_class_1_under,data_class_2_under,data_class_3_under], axis=0)
#     X,y =split_sequences(data_undersampled.values,n_steps)
#     # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
#
#     print(X.shape, y.shape)
#     X = X.reshape((X.shape[0], n_substeps, 1, n_seq, n_features))
#     print(X.shape, y.shape)
#     y = utils.np_utils.to_categorical(y)
#     np.save(filename + '_X.npy', X)
#     np.save(filename + '_y.npy', X)
#
# y_ints = [y.argmax() for y in y]
# class_weights = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
# print(class_weights)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.05, random_state=42)
# X_train = X_train.astype('float32')
# X_val = X_val.astype('float32')
# X_test = X_test.astype('float32')
# print(X_train.shape, y_train.shape,X_val.shape, y_val.shape,X_test.shape,y_test.shape)
def readdataold(filename):
    print("Creating npy files...")
    data = pandas.read_csv(filename)
    count_class_0, count_class_1, count_class_2, count_class_3, = data.Label.value_counts()
    minimum = min(count_class_0, count_class_1, count_class_2, count_class_3)
    data_class_0 = data[data['Label'] == 0]
    data_class_1 = data[data['Label'] == 1]
    data_class_2 = data[data['Label'] == 2]
    data_class_3 = data[data['Label'] == 3]
    data_class_2['Label'] = 1
    data_class_3['Label'] = 1
    # print(data_class_1.head(5))
    # print(data_class_2.head(5))
    # print(data_class_3.head(5))

    # data_class_0_under = data_class_0.sample(count_class_1)
    data_class_0_under =data_class_0.head(count_class_1)
    # data_class_1_under = data_class_1.sample(count_class_1)
    # data_class_2_under = data_class_2.sample(count_class_1)
    # data_class_3_under = data_class_3.sample(count_class_1)
    data_undersampled = pandas.concat([data_class_0_under, data_class_1], axis=0).sort_index()
    data_class_2 = pandas.concat([data_class_0_under.head(1000), data_class_2], axis=0).sort_index()
    data_class_3 = pandas.concat([data_class_0_under.head(1000), data_class_3], axis=0).sort_index()
    # print(data_undersampled.head(51890))
    # print(data_undersampled.sort_index().head(5))
    # exit(0)
    # exit(0)
    # data_undersampled = pandas.concat([data_class_0_under, data_class_1,data_class_2,data_class_3], axis=0)
    # data_undersampled = pandas.concat([data_class_0_under, data_class_1_under,data_class_2_under,data_class_3_under], axis=0)
    X, y = split_sequences(data_undersampled.values, n_steps)
    X_data_class_2, y_data_class_2 = split_sequences(data_class_2.values, n_steps)
    X_data_class_3, y_data_class_3 = split_sequences(data_class_3.values, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]

    print(X.shape, y.shape)
    print(X_data_class_2.shape, y_data_class_2.shape)
    print(X_data_class_3.shape, y_data_class_3.shape)
    X = X.reshape((X.shape[0], n_substeps, 1, n_seq, n_features))
    X_data_class_2 = X_data_class_2.reshape((X_data_class_2.shape[0], n_substeps, 1, n_seq, n_features))
    X_data_class_3 = X_data_class_3.reshape((X_data_class_3.shape[0], n_substeps, 1, n_seq, n_features))
    print(X.shape, y.shape)
    print(X_data_class_2.shape, y_data_class_2.shape)
    print(X_data_class_3.shape, y_data_class_3.shape)


    y = to_categorical(y)
    y_data_class_2 = to_categorical(y_data_class_2)
    y_data_class_3 = to_categorical(y_data_class_3)
    # np.save(filename + '_X.npy', X)
    # np.save(filename + '_y.npy', y)
    y_ints = [y.argmax() for y in y]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
    print(class_weights)
    return X,y,class_weights, X_data_class_2, y_data_class_2,X_data_class_3, y_data_class_3
def shapedataold(X,y):


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42,shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.05, random_state=42,shuffle=False)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


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
    # print(data_class_1.head(5))
    # print(data_class_2.head(5))
    # print(data_class_3.head(5))

    # data_class_0_under = data_class_0.sample(count_class_1)

    ########################class 0################################
    ########################class 0################################
    data_class_0_under = data_class_0.head(count_class_1)
    D0_train, D0_val = train_test_split(data_class_0_under, test_size=0.20, random_state=42, shuffle=False)
    D0_train, D0_test = train_test_split(D0_train, test_size=0.05, random_state=42, shuffle=False)
    print(len(D0_train), len(D0_val), len(D0_test))
    ########################class 0################################

    ########################class 1################################
    D1_train, D1_val = train_test_split(data_class_1, test_size=0.20, random_state=42, shuffle=False)
    D1_train, D1_test = train_test_split(D1_train, test_size=0.05, random_state=42, shuffle=False)
    print(len(D1_train), len(D1_val), len(D1_test))
    ########################class 1################################

    ########################class 2################################
    D2_train, D2_val = train_test_split(data_class_2, test_size=0.20, random_state=42, shuffle=False)
    D2_train, D2_test = train_test_split(D2_train, test_size=0.05, random_state=42, shuffle=False)
    print(len(D2_train), len(D2_val), len(D2_test))
    test_class_2 = pandas.concat([D0_test, D2_test], axis=0).sort_index()
    X_test2, y_test2 = split_sequences(test_class_2.values, n_steps)
    print(X_test2.shape, y_test2.shape)
    X_test2 = X_test2.reshape((X_test2.shape[0], n_substeps, 1, n_seq, n_features))
    X_test2 = X_test2.astype('float32')
    print(X_test2.shape, y_test2.shape)
    y_test2 = to_categorical(y_test2)
    ########################class 2################################

    ########################class 3################################
    D3_train, D3_val = train_test_split(data_class_3, test_size=0.20, random_state=42, shuffle=False)
    D3_train, D3_test = train_test_split(D3_train, test_size=0.05, random_state=42, shuffle=False)
    print(len(D3_train), len(D3_val), len(D3_test))
    test_class_3 = pandas.concat([D0_test, D3_test], axis=0).sort_index()
    X_test3, y_test3 = split_sequences(test_class_3.values, n_steps)
    print(X_test3.shape, y_test3.shape)
    X_test3 = X_test3.reshape((X_test3.shape[0], n_substeps, 1, n_seq, n_features))
    X_test3 = X_test3.astype('float32')
    print(X_test3.shape, y_test3.shape)
    y_test3 = to_categorical(y_test3)
    ########################class 3################################

    ########################Train Val Test Split#############################
    train_data = pandas.concat([D0_train, D1_train], axis=0).sort_index()
    val_data = pandas.concat([D0_val, D1_val], axis=0).sort_index()
    test_data = pandas.concat([D0_test, D1_test], axis=0).sort_index()
    print(len(train_data), len(val_data), len(test_data))
    ########################Train Val Test Split#############################

    ########################Split Sequences##################################
    X_train, y_train = split_sequences(train_data.values, n_steps)
    X_val, y_val = split_sequences(val_data.values, n_steps)
    X_test, y_test = split_sequences(test_data.values, n_steps)
    ########################Split Sequences##################################

    ########################Reshape Sequences##################################
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)
    # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
    X_train = X_train.reshape((X_train.shape[0], n_substeps, 1, n_seq, n_features))
    X_val = X_val.reshape((X_val.shape[0], n_substeps, 1, n_seq, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_substeps, 1, n_seq, n_features))
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)
    ########################Reshape Sequences##################################

    ########################One Hot Vector#####################################
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)
    ########################One Hot Vector#####################################

    ########################Class Weight#######################################
    y_ints = [y.argmax() for y in y_train]
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
    print(class_weights)
    ########################Class Weight#######################################
    return X_train, y_train, X_val, y_val, X_test, y_test, X_test2, y_test2, X_test3, y_test3, class_weights  # X_data_class_2, y_data_class_2, X_data_class_3, y_data_class_3
    # data_undersampled = pandas.concat([data_class_0_under, data_class_1], axis=0).sort_index()
    # data_class_2 = pandas.concat([data_class_0_under.head(1000), data_class_2], axis=0).sort_index()
    # data_class_3 = pandas.concat([data_class_0_under.head(1000), data_class_3], axis=0).sort_index()
    # # print(data_undersampled.head(51890))
    # # print(data_undersampled.sort_index().head(5))
    # # exit(0)
    # # exit(0)
    # # data_undersampled = pandas.concat([data_class_0_under, data_class_1,data_class_2,data_class_3], axis=0)
    # # data_undersampled = pandas.concat([data_class_0_under, data_class_1_under,data_class_2_under,data_class_3_under], axis=0)
    # X, y = split_sequences(data_undersampled.values, n_steps)
    # X_data_class_2, y_data_class_2 = split_sequences(data_class_2.values, n_steps)
    # X_data_class_3, y_data_class_3 = split_sequences(data_class_3.values, n_steps)

    # X = X.reshape((X.shape[0], n_substeps, 1, n_seq, n_features))
    # X_data_class_2 = X_data_class_2.reshape((X_data_class_2.shape[0], n_substeps, 1, n_seq, n_features))
    # X_data_class_3 = X_data_class_3.reshape((X_data_class_3.shape[0], n_substeps, 1, n_seq, n_features))
    # print(X.shape, y.shape)
    # print(X_data_class_2.shape, y_data_class_2.shape)
    # print(X_data_class_3.shape, y_data_class_3.shape)
    #
    # y = utils.np_utils.to_categorical(y)
    # y_data_class_2 = utils.np_utils.to_categorical(y_data_class_2)
    # y_data_class_3 = utils.np_utils.to_categorical(y_data_class_3)
    # np.save(filename + '_X.npy', X)
    # np.save(filename + '_y.npy', y)
    # y_ints = [y.argmax() for y in y]
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
    # print(class_weights)
    # return X,y,class_weights, X_data_class_2, y_data_class_2,X_data_class_3, y_data_class_3

def convlstm(X_train, y_train, X_val, y_val,class_weights):
    L1L2(l1=0.0001, l2=0.0001)
    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3,3), padding='same',activation='relu',
                         return_sequences=True, input_shape=(n_seq, 1, n_substeps, n_features),kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(ConvLSTM2D(filters=80, kernel_size=(3,3),padding='same',activation='relu', return_sequences=True,kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(ConvLSTM2D(filters=120, kernel_size=(3,3),padding='same',activation='relu',return_sequences=True,kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(ConvLSTM2D(filters=160, kernel_size=(3, 3), padding='same',activation='relu',return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(ConvLSTM2D(filters=200, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(ConvLSTM2D(filters=240, kernel_size=(3, 3), padding='same', activation='relu', return_sequences=True, kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', padding='same', data_format='channels_last', kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', padding='same', data_format='channels_last', kernel_regularizer=regularizers.L1L2(l1=0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    dt = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    checkpoint_dir = os.path.join('Checkpoint',"TLConvlstm" + dt)
    if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    checkpoints = ModelCheckpoint(
            os.path.join(checkpoint_dir, '{epoch:02d}' + '.h5'),save_weights_only=True,
            period=1)
    csv_logger = CSVLogger(os.path.join(checkpoint_dir, 'log_'+"TLConvlstm" + dt+'.csv'), append=True, separator=',')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history_callback = model.fit(X_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_data=(X_val, y_val), #class_weight=class_weights,
                                 callbacks=[checkpoints, csv_logger])
    return model

def testmodel(model, X_test, y_test):
    try:
        Y_test = np.argmax(y_test, axis=1)
        y_pred = model.predict_classes(X_test)
        print("\n", classification_report(Y_test, y_pred,digits=4))
        scores = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
    except Exception as e:
        print(e)

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, X2, y2, X3, y3, class_weights= readdata("dataset/combined_data/2nd_labeled_simple_cleaned.csv")
    trained_model = convlstm(X_train, y_train, X_val, y_val,class_weights)
    print("\nTesting Class 1...")
    testmodel(trained_model,X_test,y_test)
    print("\nTesting Class 2...")
    testmodel(trained_model, X2, y2)
    print("\nTesting Class 3...")
    testmodel(trained_model, X3, y3)

if __name__ == "__main__":
    main()
