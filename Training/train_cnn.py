from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, \
    Activation, MaxPooling2D, LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy
import tensorflow_addons as tfa


def build_cnn():
    learning_rate = 0.001
    weight_decay = 0.0001
    cnn = Sequential()

    cnn.add(Conv2D(filters=256, kernel_size=(2, 2), padding='same', input_shape=(48, 48, 3), activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU(alpha=0.1))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(rate=0.3))

    cnn.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation='relu'))
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
    cnn.add(Dense(7, activation='softmax'))
    # cnn.summary()
    cnn.compile(loss=SparseCategoricalCrossentropy(),
                optimizer=tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
                metrics=[
                    SparseCategoricalAccuracy(name="accuracy"),
                    SparseTopKCategoricalAccuracy(2, name="top-2-accuracy")]
                )
    return cnn


def train_cnn(model, train_set, validation_set):
    step_size_train = train_set.n // train_set.batch_size
    step_size_validation = validation_set.n // validation_set.batch_size
    class_weight = {0: 3995., 1: 436., 2: 4097., 3: 7215., 4: 4965., 5: 4830., 6: 3171.}
    batch_size = 256

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

    epochs = 25

    model.fit(train_set,
              steps_per_epoch=step_size_train,
              validation_data=validation_set,
              validation_steps=step_size_validation,
              batch_size=batch_size,
              epochs=epochs,
              class_weight=class_weight,
              callbacks=[checkpoint, early_stopping]
              )
