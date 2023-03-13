from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam


def build_cnn(input_shape, num_classes, learning_rate):
    cnn = Sequential()

    cnn.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same', input_shape=input_shape, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(rate=0.3))

    cnn.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(rate=0.3))

    cnn.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same', activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(rate=0.3))

    cnn.add(Flatten())
    cnn.add(Dense(512, activation='relu'))
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dense(num_classes, activation='softmax'))
    # cnn.summary()
    cnn.compile(optimizer=Adam(learning_rate=learning_rate),
                loss="categorical_crossentropy",
                metrics=["accuracy"])
    return cnn


def train_cnn(model, train_set, validation_set, class_weight, batch_size, num_epochs):
    step_size_train = train_set.n // train_set.batch_size
    step_size_validation = validation_set.n // validation_set.batch_size

    checkpoint_path = '../Models/cnn/cnn.ckpt'
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True)

    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   min_delta=0,
                                   patience=3,
                                   verbose=1,
                                   restore_best_weights=True)

    model.fit(train_set,
              steps_per_epoch=step_size_train,
              validation_data=validation_set,
              validation_steps=step_size_validation,
              batch_size=batch_size,
              epochs=num_epochs,
              class_weight=class_weight,
              callbacks=[checkpoint, early_stopping]
              )
