import os, sys
import tensorflow as tf
import time
import Discriminator, Generator, Utils


class GanModel:
    generator_training_rate = 2e-5
    descriptor_training_rate = 2e-5
    n_critic = 5
    L2_weight = 2000
    sfc_weight = 100
    vgg_weight = 100
    adv_weight = 1
    gp_weight = 10
    critic_score_weight = 1
    kernel_size = 5

    test_image_size = (2160, 3840)
    image_size = (256, 256)
    pooling_size = 2
    generator_filter_sizes = [16, 32, 64, 128, 128]
    discriminator_filter_sizes = [16, 32, 64, 128, 128]
    patch_size = 64

    gaussian_blur_multiplier = 0.0

    def __init__(self, loss_type, ground_truth, input, real, test_input_folder, results, checkpoints, tensorboard,
                 epochs, batch_size):

        self.loss_type = loss_type
        self.results = results
        self.checkpoint_dir = checkpoints
        self.epochs = epochs
        self.batch_size = batch_size
        print(f'Results will be saved to "{os.path.abspath(results)}".')
        print(f'Logs will be saved to "{os.path.abspath(tensorboard)}".')
        print(f'Checkpoints will be saved to "{os.path.abspath(checkpoints)}".')

        if self.loss_type == 'lpips':
            self.lpips_model = self.load_lpips()

        self.train_summary_writer = tf.summary.create_file_writer(tensorboard)

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.generator_training_rate, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.descriptor_training_rate, beta_1=0.5)

        self.input_loader = Utils.InputLoader(ground_truth_path=ground_truth, input_path=input, real_path=real,
                                              test_input_folder=test_input_folder)
        n_images, self.train_images, self.test_images = self.input_loader.create_image_dataset()
        if self.train_images is not None:
            self.train_images = self.train_images.batch(batch_size=self.batch_size, drop_remainder=True)
        if self.test_images is not None:
            self.test_images = self.test_images.batch(batch_size=1)

        self.n_batches = n_images // self.batch_size

        self.generator = Generator.Generator(self.pooling_size, self.generator_filter_sizes, self.kernel_size)
        self.discriminator = Discriminator.Discriminator(self.pooling_size, self.discriminator_filter_sizes,
                                                         self.kernel_size, self.patch_size)

        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_dir))

    def l2_loss(self, original_images, generated_images, mask):
        w_l2 = self.L2_weight
        mask = tf.expand_dims(mask, axis=3)
        original_images = tf.multiply(original_images, mask)
        sparse_generated_samples = tf.multiply(generated_images, mask)
        difference = tf.math.square(tf.abs(tf.subtract(original_images, sparse_generated_samples)))
        difference_mean = tf.math.sqrt(tf.reduce_sum(difference, axis=[1, 2, 3])) / (
                tf.reduce_sum(mask, axis=[1, 2, 3]) + 1e-7)
        L_l2 = tf.reduce_mean(difference_mean)
        return w_l2 * L_l2

    def load_lpips(self):
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        model = tf.keras.models.load_model('lpips_model.h5', compile=False)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        return model

    def lpips_loss(self, gen_images, orig_images):
        input0 = tf.transpose(gen_images, perm=[0, 3, 1, 2])
        input_ref = tf.transpose(orig_images, perm=[0, 3, 1, 2])

        sys.stdout = open(os.devnull, 'w')
        p0 = self.lpips_model([input0, input_ref])
        sys.stdout = sys.__stdout__
        return self.vgg_weight * tf.reduce_mean(p0)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = tf.reduce_mean(real_output)
        fake_loss = -tf.reduce_mean(fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def discriminator_train_step(self, images, n_iterations):
        ground_truth_images = images[0]
        input_images = images[1]
        real_images = images[2]
        generated_images = self.generator(input_images[:, :, :, :3])
        for _ in range(n_iterations):
            with tf.GradientTape() as discriminator_tape:
                with tf.GradientTape() as penalty_tape:
                    epsilon = tf.random.uniform(shape=(ground_truth_images.shape[0], 1, 1, 1), minval=0, maxval=1)
                    exx = epsilon * ground_truth_images + (1.0 - epsilon) * generated_images
                    penalty_tape.watch(exx)
                    gradient_value = self.discriminator(exx, training=True)

                derivative = penalty_tape.gradient(gradient_value, exx)
                penalty = tf.sqrt(tf.reduce_sum(tf.square(derivative), axis=(1, 2, 3)))
                penalty = tf.reduce_mean(tf.square(penalty - 1.0))
                penalty_loss = self.gp_weight * penalty

                real_output = self.discriminator(real_images, training=True)
                fake_output = self.discriminator(generated_images, training=True)
                critic_loss = self.critic_score_weight * self.discriminator_loss(real_output, fake_output)
                discriminator_loss = critic_loss + penalty_loss
            discriminator_gradients = discriminator_tape.gradient(discriminator_loss,
                                                                  self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(
                zip(discriminator_gradients, self.discriminator.trainable_variables))
        return critic_loss, penalty_loss

    @tf.function
    def generator_train_step(self, images, n_iterations):
        ground_truth_images = images[0]
        input_images = images[1]
        for _ in range(n_iterations):
            with tf.GradientTape() as generator_tape:
                generated_images = self.generator(input_images[:, :, :, :3], training=True)
                if self.loss_type == 'lpips':
                    perceptual_loss = self.lpips_loss(ground_truth_images, generated_images)
                else:
                    perceptual_loss = self.l2_loss(ground_truth_images, generated_images, input_images[:, :, :, 3])

                fake_output = self.discriminator(generated_images)
                adversarial_loss = self.adv_weight * tf.reduce_mean(fake_output)
                generator_loss = perceptual_loss + adversarial_loss

            generator_gradients = generator_tape.gradient(generator_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        return perceptual_loss, adversarial_loss

    def predict(self):
        self.generate_test_images(None)

    def train(self):
        k = 0
        print("Generating test images...")
        self.generate_test_images(0)
        for epoch in range(self.epochs):
            print(f'Epoch {epoch} started.')
            last_logged = time.time()

            for step, image_batch in enumerate(self.train_images):
                critic_score_loss, gradient_penalty = self.discriminator_train_step(image_batch, self.n_critic)
                perceptual_loss, adversarial_loss = self.generator_train_step(image_batch, 1)

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('Critic score', critic_score_loss, step=k)
                    tf.summary.scalar('Gradient penalty', gradient_penalty, step=k)
                    tf.summary.scalar('Perceptual loss', perceptual_loss, step=k)
                    tf.summary.scalar('Adversarial loss', adversarial_loss, step=k)

                if time.time() - last_logged > 3:
                    print(f'{step + 1} / {self.n_batches} completed.')
                    last_logged = time.time()

                k = k + 1

            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print("Generating test images...")
            self.generate_test_images(epoch + 1)

    def generate_test_images(self, epoch):
        k = 0
        for it, image in enumerate(self.test_images):
            generated_image = (self.generator(image, training=False) + 1.0) / 2.0
            generated_image = tf.cast(255 * generated_image, tf.uint8)
            generated_image = tf.squeeze(generated_image)
            generated_image = tf.image.encode_png(generated_image)
            if epoch is None:
                tf.io.write_file(f'{self.results}/image_{k}.png', generated_image)
            else:
                tf.io.write_file(f'{self.results}/image_{k}_epoch_{epoch}.png', generated_image)
            k = k + 1
