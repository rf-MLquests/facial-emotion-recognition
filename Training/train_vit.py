import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from keras import layers

num_classes = 4
input_shape = (48, 48, 3)
patch_size = 6
num_patches = 64
projection_dim = 108
batch_size = 256
num_epochs = 50
learning_rate = 0.001
weight_decay = 0.0001
num_heads = 4
num_blocks = 8


class PatchExtractor(layers.Layer):
    def __init__(self):
        super(PatchExtractor, self).__init__()

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches=num_patches, projection_dim=projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        w_init = tf.random_normal_initializer()
        class_token = w_init(shape=(1, projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches + 1, output_dim=projection_dim)

    def call(self, patch):
        batch = tf.shape(patch)[0]
        class_token = tf.tile(self.class_token, multiples=[batch, 1])
        class_token = tf.reshape(class_token, (batch, 1, self.projection_dim))
        patches_embed = self.projection(patch)
        patches_embed = tf.concat([patches_embed, class_token], 1)
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        positions_embed = self.position_embedding(positions)
        encoded = patches_embed + positions_embed
        return encoded


class MLP(layers.Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.dense1 = layers.Dense(hidden_features, activation=tf.nn.gelu)
        self.dense2 = layers.Dense(out_features)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        y = self.dropout(x)
        return y


class Block(layers.Layer):
    def __init__(self, projection_dim, num_heads=num_heads, dropout_rate=0.1):
        super(Block, self).__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(projection_dim * 2, projection_dim, dropout_rate)

    def call(self, x):
        x1 = self.norm1(x)
        attention_output = self.attn(x1, x1)
        x2 = layers.Add()([attention_output, x])
        x3 = self.norm2(x2)
        x3 = self.mlp(x3)
        y = layers.Add()([x3, x2])
        return y


class TransformerEncoder(layers.Layer):
    def __init__(self, projection_dim, num_heads=num_heads, num_blocks=num_blocks, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.blocks = [Block(projection_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(0.5)

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        y = self.dropout(x)
        return y


def build_vit():
    inputs = layers.Input(shape=input_shape)
    patches = PatchExtractor()(inputs)
    patches_embed = PatchEncoder(num_patches, projection_dim)(patches)
    representation = TransformerEncoder(projection_dim)(patches_embed)
    representation = layers.GlobalAveragePooling1D()(representation)
    logits = MLP(projection_dim, num_classes, 0.5)(representation)
    model = keras.Model(inputs=inputs, outputs=logits)
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(2, name="top-2-accuracy"),
        ],
    )
    return model


def train_vit(model, train_set, validation_set):
    step_size_train = train_set.n // train_set.batch_size
    step_size_validation = validation_set.n // validation_set.batch_size
    checkpoint_callback = keras.callbacks.ModelCheckpoint("../Models/vit/vit.ckpt",
                                                          monitor="val_accuracy",
                                                          save_best_only=True,
                                                          save_weights_only=True
                                                          )

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                   min_delta=0,
                                                   patience=3,
                                                   verbose=1,
                                                   restore_best_weights=True
                                                   )

    model.fit(train_set,
              steps_per_epoch=step_size_train,
              validation_data=validation_set,
              validation_steps=step_size_validation,
              batch_size=batch_size,
              epochs=num_epochs,
              callbacks=[early_stopping, checkpoint_callback])
