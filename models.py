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
        # Pass the source and target images through the generators
        first_image = self.first_generator(source_img)
        second_image = self.second_generator(target_img)

        return first_image, second_image

    def calc_total_loss(self, first_image, second_image, source_image, target_image):
        # Calculate the generator and discriminator losses
        # Generator losses
        first_gen_loss = cycle_loss(self.second_generator, self.first_discriminator, first_image, source_image)
        second_gen_loss = cycle_loss(self.first_generator, self.second_discriminator, second_image, target_image)

        # Identity losses
        first_identity_loss = identity_loss(self.first_generator, source_image)
        second_identity_loss = identity_loss(self.second_generator, target_image)

        # Total generator loss
        generator_loss = first_gen_loss + second_gen_loss + first_identity_loss + second_identity_loss

        # Discriminator losses
        first_disc_loss = discriminator_loss(self.first_discriminator, first_image, source_image)
        second_disc_loss = discriminator_loss(self.second_discriminator, second_image, target_image)

        # Total discriminator loss
        discriminator_loss = first_disc_loss + second_disc_loss

        return generator_loss, discriminator_loss

    def train(self, source_image, target_image):

        optimizer = tf.keras.optimizers.Adam()

        for epoch in range(10):
            with tf.GradientTape(persistent=True) as tape:
                first_image, second_image = self.calc_outputs(source_image, target_image)
                generator_loss, discriminator_loss = self.calc_total_loss(
                    first_image, second_image, source_image, target_image
                )

            # Compute the gradients
            generator_gradients = tape.gradient(generator_loss, self.trainable_variables)
            discriminator_gradients = tape.gradient(discriminator_loss, self.trainable_variables)

            # Apply the gradients to update the model's weights
            optimizer.apply_gradients(zip(generator_gradients, self.trainable_variables))
            optimizer.apply_gradients(zip(discriminator_gradients, self.trainable_variables))

            # Print the losses for monitoring
            print(f"Epoch {epoch + 1}: Generator Loss: {generator_loss}, Discriminator Loss: {discriminator_loss}")




    