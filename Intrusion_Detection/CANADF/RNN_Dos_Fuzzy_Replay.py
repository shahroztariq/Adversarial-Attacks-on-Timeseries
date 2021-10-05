import keras
from keras import Sequential, regularizers
from keras.layers import Dense, BatchNormalization, Dropout, TimeDistributed, LSTM, LayerNormalization
from scipy.stats import mode
import pandas
import numpy as np

# Training parameters.
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

# from layer_norm import LayerNorm1D
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0],True)

batch_size = 128 # 16 no good
num_classes = 4
epochs = 20
first_dim = 40
# Number of features
feature_idx = 11
print('loading data')
dataframe = pandas.read_csv("dataset/combined_data/2nd_labeled.csv", header=None)
# dataframe = pandas.read_csv("dataset/combined_data/test.csv", header=None)

dataset = dataframe.values
print('data loaded')
print('Making Train Validation Test Set')
datasetlen=len(dataset)
# print(datasetlen % 10)
if datasetlen % first_dim is not 0:
    datasetlen = datasetlen-datasetlen % first_dim
X = np.array([dataset[i:i+first_dim, 0:feature_idx].astype('float32') for i in range(0, datasetlen, first_dim)])
Y = np.array([mode(dataset[i:i+first_dim,feature_idx:].astype(int))[0][0][0] for i in range(0, datasetlen, first_dim)])

#cann't do this because it will split all the patterns in car
#print(X[0])
print(set(Y))
X=X.reshape(X.shape[0], first_dim, feature_idx, 1)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=42)
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

y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)
print('Making Model')
model=Sequential()
model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.0001), input_shape=(first_dim, feature_idx,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(TimeDistributed(LSTM(512,kernel_regularizer=regularizers.l2(0.0001))))
model.add(LayerNormalization())
model.add(LSTM(512,kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.0001)))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# Training.
history_callback = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val))

model.save('CAN_RNN_model_1line_2021.h5')


model.load_weights('CAN_RNN_model_1line_2021.h5')


Y_test = np.argmax(y_test, axis=1)
y_pred = model.predict_classes(X_test)

print("\n", classification_report(Y_test, y_pred))

scores = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])