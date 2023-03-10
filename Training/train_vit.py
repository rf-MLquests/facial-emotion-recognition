import tensorflow_addons as tfa
from keras import layers
import tensorflow as tf
from tensorflow import keras
from keras import layers

input_shape = (48, 48, 3)
num_classes = 7
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 40
image_size = 48
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 108
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 8
mlp_head_units = [2048, 1024]


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def build_vit():
    inputs = layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[
                      keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                      keras.metrics.SparseTopKCategoricalAccuracy(2, name='top-2-accuracy')]
                  )
    return model


def train_vit(model, train_set, validation_set):
    step_size_train = train_set.n // train_set.batch_size
    step_size_validation = validation_set.n // validation_set.batch_size
    class_weight = {0: 3995., 1: 436., 2: 4097., 3: 7215., 4: 4965., 5: 4830., 6: 3171.}
    checkpoint_path = '../Models/vit/vit.ckpt'

    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 monitor='val_accuracy',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                   min_delta=0,
                                                   patience=4,
                                                   verbose=1,
                                                   restore_best_weights=True)

    epochs = 40

    model.fit(train_set,
              steps_per_epoch=step_size_train,
              validation_data=validation_set,
              validation_steps=step_size_validation,
              batch_size=batch_size,
              epochs=epochs,
              class_weight=class_weight,
              callbacks=[checkpoint, early_stopping]
              )
