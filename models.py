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

class Discriminator(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.discriminator_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 4, stride=2, padding=1, activation="relu"),
            tf.keras.layers.Conv2D(128, 4, stride=2, padding=1, activation="relu"),
            tf.keras.layers.Conv2D(256, 4, stride=2, padding=1, activation="relu"),
            tf.keras.layers.Conv2D(512, 4, stride=1, padding=1, activation="relu"),
            tf.keras.layers.Conv2D(1, 4, stride=1, padding=1, activation="sigmoid"),
        ])

    def call(self, inputs):

        outputs = self.discriminator_block(inputs)

        return outputs
    
class CycleGAN(tf.keras.Model):

    def __init__(self):

        super().__init__()

        self.first_generator = Generator()
        self.second_generator = Generator()

        self.first_discriminator = Discriminator()
        self.second_discriminator = Discriminator()

    def calc_outputs(self, source_img, target_img):

        pass

    def calc_total_loss(self, first_image, second_image, source_image, target_image):

        pass

    def train(self, source_image, target_image):

        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(epochs):

            with tf.GradientTape(persistent=True) as tape:

                first_image, second_image = self.calc_outputs(source_image, target_image)

                loss = self.calc_total_loss(first_image, second_image, source_image, target_image)



    