import pandas
import numpy as np
import math
import statistics
import keras.backend as K
from time import gmtime, strftime
from keras import applications, Sequential, utils, regularizers
from keras.layers import Dense, BatchNormalization, Dropout, TimeDistributed, LSTM, Flatten, Conv2D, MaxPooling2D, ConvLSTM2D, LayerNormalization
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint, CSVLogger
from layer_norm import LayerNorm1D
import os
# dtype='float16'
# K.set_floatx(dtype)
#
# # default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
# K.set_epsilon(1e-4)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
feature_size=11
window_size=32
batch_size=128
num_classes=4
epochs=50

# class_weight = {0: 1.,
#                 1: 40.,
#                 2: 25.,
#                 3:50}
data=pandas.read_csv("dataset/combined_data/2nd_labeled_simple_cleaned.csv")
# data=pandas.read_csv("dataset/combined_data/test.csv")

label=data['Label']
# print(Label.head(10))
data= data.drop("Label", axis=1)
useful_rows=len(data)-((len(data))%window_size)
total_windows=int(useful_rows/window_size)
print(useful_rows,total_windows)
print(data.head(useful_rows).values.reshape(total_windows,feature_size,window_size).shape)
X=data.head(useful_rows).values.reshape(total_windows,feature_size,window_size,1)
y=label.head(useful_rows).values.reshape(total_windows,1,window_size)
y=np.array([int(statistics.mean(i[0])) for i in y]).reshape(-1,1)

y=utils.np_utils.to_categorical(y)
y_ints = [y.argmax() for y in y]
class_weights=class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)
print(class_weights)
# print(y)
print(X.shape)
print(y.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.05, random_state=42)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

model=Sequential()
model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.0001), input_shape=(feature_size, window_size,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(TimeDistributed(LSTM(128,kernel_regularizer=regularizers.l2(0.0001))))
model.add(LayerNormalization())
model.add(LSTM(256,kernel_regularizer=regularizers.l2(0.0001)))
# model.add(Flatten())
model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.0001)))
model.summary()



# checkpoint_dir= os.path.join('Checkpoint', strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
# if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
# checkpoints = ModelCheckpoint(
#         os.path.join(checkpoint_dir, '{epoch:02d}' + '.h5'),save_weights_only=True,
#         period=1)
# csv_logger = CSVLogger(os.path.join(checkpoint_dir, 'log.csv'), append=True, separator=',')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# # Training.
# history_callback = model.fit(X_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(X_val, y_val),class_weight=class_weights,
#          callbacks=[checkpoints, csv_logger])
# final_model=os.path.join('Checkpoint', strftime("%Y-%m-%d-%H-%M-%S", gmtime()),'CAN19_RNN_model.h5')
# model.save(final_model)

# model.load_weights(final_model)
Y_test = np.argmax(y_test, axis=1)
y_pred = model.predict_classes(X_test)

print("\n", classification_report(Y_test, y_pred))

scores = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])