import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):

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

    