import tensorflow_addons as tfa
import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, Input, Dense, Dropout, Embedding, MultiHeadAttention, LayerNormalization, Add, Flatten
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy
from keras.callbacks import ModelCheckpoint, EarlyStopping


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x


class Patches(Layer):
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


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def build_vit(input_shape, num_classes, patch_size, num_heads, transformer_layers, learning_rate, weight_decay):
    image_size = input_shape[0]
    num_patches = (image_size // patch_size) ** 2
    projection_dim = (patch_size ** 2) * input_shape[2]
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    mlp_head_units = [2048, 1024]
    print("-----DEBUG-----")
    print("number of patches: " + num_patches)
    print("projection dimension: " + projection_dim)

    inputs = Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = Add()([x3, x2])

    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = Dense(num_classes)(features)
    model = Model(inputs=inputs, outputs=logits)

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(optimizer=optimizer,
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[
                      SparseCategoricalAccuracy(name="accuracy"),
                      SparseTopKCategoricalAccuracy(2, name='top-2-accuracy')]
                  )
    return model


def train_vit(model, train_set, validation_set, class_weight, batch_size, num_epochs):
    step_size_train = train_set.n // train_set.batch_size
    step_size_validation = validation_set.n // validation_set.batch_size
    checkpoint_path = '../Models/vit/vit.ckpt'

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
