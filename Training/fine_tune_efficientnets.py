from keras.applications import EfficientNetB0, EfficientNetV2B0
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


def build_efficientnet(input_shape, num_classes, learning_rate, use_v2=True):
    inputs = Input(shape=input_shape)
    efficientnet = None
    if use_v2:
        print("Using EfficientNet V2")
        efficientnet = EfficientNetV2B0(include_top=False, input_tensor=inputs, weights="imagenet")
    else:
        print("Using EfficientNet")
        efficientnet = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    efficientnet.trainable = False

    x = GlobalAveragePooling2D(name="avg_pool")(efficientnet.output)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)

    efficientnet = Model(inputs, outputs, name="EfficientNet")
    optimizer = Adam(learning_rate=learning_rate)
    efficientnet.summary()
    efficientnet.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # efficientnet.summary()
    return efficientnet


def train_efficientnet(model, train_set, validation_set, class_weight, batch_size, num_epochs, use_v2=True):
    step_size_train = train_set.n // train_set.batch_size
    step_size_validation = validation_set.n // validation_set.batch_size
    checkpoint_path = None
    if use_v2:
        print("Using EfficientNet V2")
        checkpoint_path = '../Models/efficientnetv2/efficientnet.ckpt'
    else:
        print("Using EfficientNet")
        checkpoint_path = '../Models/efficientnet/efficientnet.ckpt'

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
