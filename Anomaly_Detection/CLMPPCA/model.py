#X_train shape = (S,T,h,f,i) see figure 4 in the paper
def KARI_MODEL_convlstm(X_train,n_classes):
    model = tf.keras.Sequential()
    print(X_train.shape)
    model.add(tf.keras.layers.ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3],X_train.shape[4]),
                       padding='same', return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.ConvLSTM2D(filters=80, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.ConvLSTM2D(filters=120, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.ConvLSTM2D(filters=160, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(n_classes, activation='linear'))
    return model