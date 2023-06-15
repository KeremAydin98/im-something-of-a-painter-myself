import tensorflow as tf


class ResidualLayer(tf.keras.layers.Layer):

    def __init__(self):

        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(256, 3)

        self.batch_norm = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(256, 3)

        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):

        x = self.conv1(inputs)

        x = self.batch_norm(x)

        x = self.relu(x)

        x = self.conv2(x)

        x = self.batch_norm_2(x)

        outputs = tf.keras.layers.add([x, inputs])

        return outputs
    
class Generator(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.first_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 7, stride=1, padding=3, activation="relu"),
            tf.keras.layers.Conv2D(128, 3, stride=2, padding=1, activation="relu"),
            tf.keras.layers.Conv2D(256, 3, stride=2, padding=1, activation="relu")
        ])

        self.residualblock = tf.keras.Sequential([
            ResidualLayer(),
            ResidualLayer(),
            ResidualLayer(),
            ResidualLayer(),
            ResidualLayer(),
            ResidualLayer(),
            ResidualLayer(),
            ResidualLayer(),
            ResidualLayer(),
        ])

        self.output_block = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(128, 3, stride=2, padding=1, out_padding=1, activation="relu"),
            tf.keras.layers.Conv2DTranspose(64, 3, stride=2, padding=1, out_padding=1, activation="relu"),
            tf.keras.layers.Conv2D(3, 7, stride=1, padding=3)
        ])

    def call(self, inputs):

        x = self.first_block(inputs)

        x = self.residualblock(x)

        outputs = self.output_block(x)

        return outputs
