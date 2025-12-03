import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, Conv2D, Conv2DTranspose, Concatenate, \
    MultiHeadAttention, MaxPooling2D, Cropping2D, Dense, LayerNormalization, Permute


@tf.keras.saving.register_keras_serializable()
class ConvResBlock(tf.keras.layers.Layer):
    def __init__(self, channel, kernel_size, dropout_rate=0.0, norm_layer=BatchNormalization, activation='relu',
                 **kwargs):
        super().__init__(**kwargs)
        self.norm1 = norm_layer(name='norm1')
        self.act1 = Activation(activation, name='act1')
        self.conv1 = Conv2D(channel, kernel_size=kernel_size, strides=(1, 1), padding='same', name='conv1')
        self.norm2 = norm_layer(name='norm2')
        self.act2 = Activation(activation, name='act2')
        self.drop = Dropout(dropout_rate, name='drop2')
        self.conv2 = Conv2D(channel, kernel_size=kernel_size, strides=(1, 1), padding='same', name='conv2')

    def call(self, inputs):
        skip = inputs
        outputs = self.norm1(inputs)
        outputs = self.act1(outputs)
        outputs = self.conv1(outputs)
        outputs = self.norm2(outputs)
        outputs = self.act2(outputs)
        outputs = self.drop(outputs)
        outputs = self.conv2(outputs)
        outputs += skip
        return outputs


@tf.keras.saving.register_keras_serializable()
class LatentArrayInitialization(tf.keras.layers.Layer):
    def __init__(self, num_latents, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latents = self.add_weight(shape=(num_latents, latent_dim), initializer='random_normal',
                                       trainable=True, name='lat')

    def call(self, inputs):
        B, T = tf.shape(inputs)[0], tf.shape(inputs)[1]
        latents = tf.expand_dims(self.latents, axis=0)
        latents = tf.repeat(latents, repeats=T, axis=0)
        latents = tf.expand_dims(latents, axis=0)
        latents = tf.repeat(latents, repeats=B, axis=0)
        outputs = latents
        return outputs


@tf.keras.saving.register_keras_serializable()
class CrossTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.0, norm_layer=LayerNormalization, activation='relu',
                 **kwargs):
        super().__init__(**kwargs)
        self.norm1 = norm_layer(name='norm1')
        self.norm0 = norm_layer(name='norm0')
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name='attn')
        self.drop1 = Dropout(dropout_rate, name='drop1')
        self.norm2 = norm_layer(name='norm2')
        self.act = Activation(activation, name='act')
        self.drop2 = Dropout(dropout_rate, name='drop2')
        self.drop3 = Dropout(dropout_rate, name='drop3')

    def build(self, input_shape):
        channels = input_shape[0][-1]
        self.linear1 = Dense(channels * 4, name='fc1')
        self.linear2 = Dense(channels, name='fc2')

    def call(self, inputs):
        query, value = inputs
        B, T, N, D = query.shape
        query = tf.reshape(query, [-1, N, D])
        _, _, F, C = value.shape
        value = tf.reshape(value, [-1, F, C])
        skip = query
        norm_q = self.norm1(query)
        norm_v = self.norm0(value)
        attn_outputs = self.attn(norm_q, norm_v)
        attn_outputs = self.drop1(attn_outputs)
        attn_outputs += skip
        outputs = self.norm2(attn_outputs)
        outputs = self.linear1(outputs)
        outputs = self.act(outputs)
        outputs = self.drop2(outputs)
        outputs = self.linear2(outputs)
        outputs = self.drop3(outputs)
        outputs += attn_outputs
        outputs = tf.reshape(outputs, [-1, T, N, D])
        return outputs


@tf.keras.saving.register_keras_serializable()
class TimeTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.0, norm_layer=LayerNormalization, activation='relu',
                 **kwargs):
        super().__init__(**kwargs)
        self.norm1 = norm_layer(name='norm1')
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name='attn')
        self.drop1 = Dropout(dropout_rate, name='drop1')
        self.norm2 = norm_layer(name='norm2')
        self.act = Activation(activation, name='act')
        self.drop2 = Dropout(dropout_rate, name='drop2')
        self.drop3 = Dropout(dropout_rate, name='drop3')

    def build(self, input_shape):
        channels = input_shape[-1]
        self.linear1 = Dense(channels * 4, name='fc1')
        self.linear2 = Dense(channels, name='fc2')

    def call(self, inputs):
        B, T, N, D = inputs.shape
        inputs = tf.transpose(inputs, [0, 2, 1, 3])
        inputs = tf.reshape(inputs, [-1, T, D])
        skip = inputs
        attn_outputs = self.norm1(inputs)
        attn_outputs = self.attn(attn_outputs, attn_outputs)
        attn_outputs = self.drop1(attn_outputs)
        attn_outputs += skip
        outputs = self.norm2(attn_outputs)
        outputs = self.linear1(outputs)
        outputs = self.act(outputs)
        outputs = self.drop2(outputs)
        outputs = self.linear2(outputs)
        outputs = self.drop3(outputs)
        outputs += attn_outputs
        outputs = tf.reshape(outputs, [-1, N, T, D])
        outputs = tf.transpose(outputs, [0, 2, 1, 3])
        return outputs


@tf.keras.saving.register_keras_serializable()
class LatentTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.0, norm_layer=LayerNormalization, activation='relu',
                 **kwargs):
        super().__init__(**kwargs)
        self.norm1 = norm_layer(name='norm1')
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name='attn')
        self.drop1 = Dropout(dropout_rate, name='drop1')
        self.norm2 = norm_layer(name='norm2')
        self.act = Activation(activation, name='act')
        self.drop2 = Dropout(dropout_rate, name='drop2')
        self.drop3 = Dropout(dropout_rate, name='drop3')

    def build(self, input_shape):
        channels = input_shape[-1]
        self.linear1 = Dense(channels * 4, name='fc1')
        self.linear2 = Dense(channels, name='fc2')

    def call(self, inputs):
        B, T, N, D = inputs.shape
        inputs = tf.reshape(inputs, [-1, N, D])
        skip = inputs
        attn_outputs = self.norm1(inputs)
        attn_outputs = self.attn(attn_outputs, attn_outputs)
        attn_outputs = self.drop1(attn_outputs)
        attn_outputs += skip
        outputs = self.norm2(attn_outputs)
        outputs = self.linear1(outputs)
        outputs = self.act(outputs)
        outputs = self.drop2(outputs)
        outputs = self.linear2(outputs)
        outputs = self.drop3(outputs)
        outputs += attn_outputs
        outputs = tf.reshape(outputs, [-1, T, N, D])
        return outputs


def oafs_model(feature_num, timesteps, ch_num, out_class, dropout_rate, transformer_dropout_rate, start, end,
               norm_layer=BatchNormalization, transformer_norm_layer=LayerNormalization, activation='relu'):
    input_score = Input(shape=(timesteps, feature_num, ch_num))
    en = Conv2D(2 ** 6, kernel_size=(3, 3), strides=(1, 1), padding="same")(input_score)

    en_l1 = ConvResBlock(2 ** 6, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en)
    en_l1 = ConvResBlock(2 ** 6, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l1)
    en_l1 = norm_layer()(en_l1)
    en_l1 = Activation(activation)(en_l1)
    en_l1 = Conv2D(2 ** 7, kernel_size=(3, 3), strides=(2, 2), padding="same")(en_l1)

    en_l2 = ConvResBlock(2 ** 7, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l1)
    en_l2 = ConvResBlock(2 ** 7, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l2)
    en_l2 = norm_layer()(en_l2)
    en_l2 = Activation(activation)(en_l2)
    en_l2 = Conv2D(2 ** 8, kernel_size=(3, 3), strides=(2, 2), padding="same")(en_l2)

    en_l3 = ConvResBlock(2 ** 8, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l2)
    en_l3 = ConvResBlock(2 ** 8, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l3)
    en_l3 = norm_layer()(en_l3)
    en_l3 = Activation(activation)(en_l3)
    en_l3 = Conv2D(2 ** 9, kernel_size=(3, 3), strides=(2, 2), padding="same")(en_l3)

    en_l4 = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l3)
    en_l4 = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l4)
    en_l4 = norm_layer()(en_l4)
    en_l4 = Activation(activation)(en_l4)
    en_l4 = Conv2D(2 ** 9, kernel_size=(3, 3), strides=(2, 2), padding="same")(en_l4)

    bottle = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l4)
    bottle = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(bottle)

    de_l1 = Concatenate()([bottle, en_l4])
    de_l1 = norm_layer()(de_l1)
    de_l1 = Activation(activation)(de_l1)
    de_l1 = Conv2D(2 ** 9, kernel_size=(3, 3), strides=(1, 1), padding="same")(de_l1)
    de_l1 = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l1)
    de_l1 = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l1)
    de_l1 = norm_layer()(de_l1)
    de_l1 = Activation(activation)(de_l1)
    de_l1 = Conv2DTranspose(2 ** 9, kernel_size=(3, 3), strides=(2, 2), padding="same")(de_l1)

    de_l2 = Concatenate()([de_l1, en_l3])
    de_l2 = norm_layer()(de_l2)
    de_l2 = Activation(activation)(de_l2)
    de_l2 = Conv2D(2 ** 9, kernel_size=(3, 3), strides=(1, 1), padding="same")(de_l2)
    de_l2 = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l2)
    de_l2 = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l2)
    de_l2 = norm_layer()(de_l2)
    de_l2 = Activation(activation)(de_l2)
    de_l2 = Conv2DTranspose(2 ** 8, kernel_size=(3, 3), strides=(2, 2), padding="same")(de_l2)

    de_l3 = Concatenate()([de_l2, en_l2])
    de_l3 = norm_layer()(de_l3)
    de_l3 = Activation(activation)(de_l3)
    de_l3 = Conv2D(2 ** 8, kernel_size=(3, 3), strides=(1, 1), padding="same")(de_l3)
    de_l3 = ConvResBlock(2 ** 8, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l3)
    de_l3 = ConvResBlock(2 ** 8, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l3)
    de_l3 = norm_layer()(de_l3)
    de_l3 = Activation(activation)(de_l3)
    de_l3 = Conv2DTranspose(2 ** 7, kernel_size=(3, 3), strides=(2, 2), padding="same")(de_l3)

    de_l4 = Concatenate()([de_l3, en_l1])
    de_l4 = norm_layer()(de_l4)
    de_l4 = Activation(activation)(de_l4)
    de_l4 = Conv2D(2 ** 7, kernel_size=(3, 3), strides=(1, 1), padding="same")(de_l4)
    de_l4 = ConvResBlock(2 ** 7, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l4)
    de_l4 = ConvResBlock(2 ** 7, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l4)
    de_l4 = norm_layer()(de_l4)
    de_l4 = Activation(activation)(de_l4)
    de_l4 = Conv2DTranspose(2 ** 6, kernel_size=(3, 3), strides=(2, 2), padding="same")(de_l4)

    de_out = LatentArrayInitialization(out_class * 2, 128)(de_l4)
    de_out = CrossTransformerBlock(num_heads=8, key_dim=32, dropout_rate=transformer_dropout_rate,
                                   norm_layer=transformer_norm_layer)([de_out, de_l4])
    de_out = TimeTransformerBlock(num_heads=8, key_dim=32, dropout_rate=transformer_dropout_rate,
                                  norm_layer=transformer_norm_layer)(de_out)
    de_out = LatentTransformerBlock(num_heads=8, key_dim=32, dropout_rate=transformer_dropout_rate,
                                    norm_layer=transformer_norm_layer)(de_out)
    de_out = CrossTransformerBlock(num_heads=8, key_dim=32, dropout_rate=transformer_dropout_rate,
                                   norm_layer=transformer_norm_layer)([de_out, de_l4])
    de_out = TimeTransformerBlock(num_heads=8, key_dim=32, dropout_rate=transformer_dropout_rate,
                                  norm_layer=transformer_norm_layer)(de_out)
    de_out = LatentTransformerBlock(num_heads=8, key_dim=32, dropout_rate=transformer_dropout_rate,
                                    norm_layer=transformer_norm_layer)(de_out)
    de_out = CrossTransformerBlock(num_heads=8, key_dim=32, dropout_rate=transformer_dropout_rate,
                                   norm_layer=transformer_norm_layer)([de_out, de_l4])
    de_out = TimeTransformerBlock(num_heads=8, key_dim=32, dropout_rate=transformer_dropout_rate,
                                  norm_layer=transformer_norm_layer)(de_out)
    de_out = LatentTransformerBlock(num_heads=8, key_dim=32, dropout_rate=transformer_dropout_rate,
                                    norm_layer=transformer_norm_layer)(de_out)

    onset_out = Cropping2D(cropping=((0, 0), (0, out_class)))(de_out)
    onset_out = (Conv2D(end - start + 1, (1, 1), strides=(1, 1), padding="same", activation="sigmoid")
                 (onset_out))
    frame_out = Cropping2D(cropping=((0, 0), (out_class, 0)))(de_out)
    frame_out = Concatenate()([frame_out, onset_out])
    frame_out = (Conv2D(end - start + 1, (1, 1), strides=(1, 1), padding="same", activation="sigmoid")
                 (frame_out))
    onset_out = Permute((1, 3, 2))(onset_out)
    frame_out = Permute((1, 3, 2))(frame_out)
    onset_out = Cropping2D(cropping=((0, 0), (start, 127 - end)), name="onset")(onset_out)
    frame_out = Cropping2D(cropping=((0, 0), (start, 127 - end)), name="frame")(frame_out)
    out = [onset_out, frame_out]
    return Model(inputs=input_score, outputs=out)


def oafs_resunet(feature_num, timesteps, ch_num, out_class, dropout_rate, transformer_dropout_rate, start, end,
                 norm_layer=BatchNormalization, transformer_norm_layer=LayerNormalization, activation='relu'):
    input_score = Input(shape=(timesteps, feature_num, ch_num))
    en = Conv2D(2 ** 6, kernel_size=(3, 3), strides=(1, 1), padding="same")(input_score)

    en_l1 = ConvResBlock(2 ** 6, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en)
    en_l1 = ConvResBlock(2 ** 6, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l1)
    en_l1 = norm_layer()(en_l1)
    en_l1 = Activation(activation)(en_l1)
    en_l1 = Conv2D(2 ** 7, kernel_size=(3, 3), strides=(2, 2), padding="same")(en_l1)

    en_l2 = ConvResBlock(2 ** 7, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l1)
    en_l2 = ConvResBlock(2 ** 7, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l2)
    en_l2 = norm_layer()(en_l2)
    en_l2 = Activation(activation)(en_l2)
    en_l2 = Conv2D(2 ** 8, kernel_size=(3, 3), strides=(2, 2), padding="same")(en_l2)

    en_l3 = ConvResBlock(2 ** 8, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l2)
    en_l3 = ConvResBlock(2 ** 8, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l3)
    en_l3 = norm_layer()(en_l3)
    en_l3 = Activation(activation)(en_l3)
    en_l3 = Conv2D(2 ** 9, kernel_size=(3, 3), strides=(2, 2), padding="same")(en_l3)

    en_l4 = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l3)
    en_l4 = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l4)
    en_l4 = norm_layer()(en_l4)
    en_l4 = Activation(activation)(en_l4)
    en_l4 = Conv2D(2 ** 9, kernel_size=(3, 3), strides=(2, 2), padding="same")(en_l4)

    bottle = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(en_l4)
    bottle = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(bottle)

    de_l1 = Concatenate()([bottle, en_l4])
    de_l1 = norm_layer()(de_l1)
    de_l1 = Activation(activation)(de_l1)
    de_l1 = Conv2D(2 ** 9, kernel_size=(3, 3), strides=(1, 1), padding="same")(de_l1)
    de_l1 = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l1)
    de_l1 = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l1)
    de_l1 = norm_layer()(de_l1)
    de_l1 = Activation(activation)(de_l1)
    de_l1 = Conv2DTranspose(2 ** 9, kernel_size=(3, 3), strides=(2, 2), padding="same")(de_l1)

    de_l2 = Concatenate()([de_l1, en_l3])
    de_l2 = norm_layer()(de_l2)
    de_l2 = Activation(activation)(de_l2)
    de_l2 = Conv2D(2 ** 9, kernel_size=(3, 3), strides=(1, 1), padding="same")(de_l2)
    de_l2 = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l2)
    de_l2 = ConvResBlock(2 ** 9, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l2)
    de_l2 = norm_layer()(de_l2)
    de_l2 = Activation(activation)(de_l2)
    de_l2 = Conv2DTranspose(2 ** 8, kernel_size=(3, 3), strides=(2, 2), padding="same")(de_l2)

    de_l3 = Concatenate()([de_l2, en_l2])
    de_l3 = norm_layer()(de_l3)
    de_l3 = Activation(activation)(de_l3)
    de_l3 = Conv2D(2 ** 8, kernel_size=(3, 3), strides=(1, 1), padding="same")(de_l3)
    de_l3 = ConvResBlock(2 ** 8, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l3)
    de_l3 = ConvResBlock(2 ** 8, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l3)
    de_l3 = norm_layer()(de_l3)
    de_l3 = Activation(activation)(de_l3)
    de_l3 = Conv2DTranspose(2 ** 7, kernel_size=(3, 3), strides=(2, 2), padding="same")(de_l3)

    de_l4 = Concatenate()([de_l3, en_l1])
    de_l4 = norm_layer()(de_l4)
    de_l4 = Activation(activation)(de_l4)
    de_l4 = Conv2D(2 ** 7, kernel_size=(3, 3), strides=(1, 1), padding="same")(de_l4)
    de_l4 = ConvResBlock(2 ** 7, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l4)
    de_l4 = ConvResBlock(2 ** 7, kernel_size=(3, 3), dropout_rate=dropout_rate, norm_layer=norm_layer)(de_l4)
    de_l4 = norm_layer()(de_l4)
    de_l4 = Activation(activation)(de_l4)
    de_l4 = Conv2DTranspose(2 ** 6, kernel_size=(3, 3), strides=(2, 2), padding="same")(de_l4)

    de_out = de_l4
    onset_out = Conv2D(out_class, (1, 1), strides=(1, 1), padding="same", activation="sigmoid")(de_out)
    frame_out = Concatenate()([de_out, onset_out])
    frame_out = Conv2D(out_class, (1, 1), strides=(1, 1), padding="same", activation="sigmoid")(frame_out)
    onset_out = MaxPooling2D(pool_size=(1, feature_num // 128))(onset_out)
    onset_out = Cropping2D(cropping=((0, 0), (start, 127 - end)), name="onset")(onset_out)
    frame_out = MaxPooling2D(pool_size=(1, feature_num // 128))(frame_out)
    frame_out = Cropping2D(cropping=((0, 0), (start, 127 - end)), name="frame")(frame_out)
    out = [onset_out, frame_out]
    return Model(inputs=input_score, outputs=out)