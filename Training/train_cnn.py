from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, \
    Activation, MaxPooling2D, LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def build_cnn():
    cnn = Sequential()

    cnn.add(Conv2D(filters=256, kernel_size=(2, 2), padding='same', input_shape=(48, 48, 3), activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(rate=0.2))

    cnn.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(rate=0.2))

    cnn.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(rate=0.2))

    cnn.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(rate=0.2))

    cnn.add(Flatten())
    cnn.add(Dense(512, activation='relu'))
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dense(4, activation='softmax'))
    # cnn.summary()
    return cnn


def compile_and_train(model, train_set, validation_set):
    checkpoint = ModelCheckpoint("../Models/CNN_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')

    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   min_delta=0,
                                   patience=3,
                                   verbose=1,
                                   restore_best_weights=True
                                   )

    reduce_learningrate = ReduceLROnPlateau(monitor='val_accuracy',
                                            factor=0.2,
                                            patience=3,
                                            verbose=1,
                                            min_delta=0.0001)

    callbacks_list = [early_stopping, checkpoint, reduce_learningrate]
    epochs = 20
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    history = model.fit(train_set, validation_data=validation_set, epochs=epochs, callbacks=callbacks_list, verbose=1)
    return history


def evaluate_model(model, test_set):
    model.evaluate(test_set)
