import tensorflow as tf
from omegaconf import DictConfig

from .asr_model import build_w2l_model, LogMel

tfkl = tf.keras.layers


class ConversionModel(tf.keras.Model):
    def __init__(self, inputs, outputs,
                 content_model,
                 speaker_classification_model,
                 gradient_clipping, **kwargs):
        super().__init__(inputs, outputs, **kwargs)
        self.loss_tracker = tf.metrics.Mean(name="loss")
        self.speaker_confusion_tracker = tf.metrics.Mean(name="confusion_loss")

        self.speaker_loss_real_tracker = tf.metrics.Mean(name="speaker_loss_real")
        self.speaker_accuracy_real_tracker = tf.metrics.BinaryAccuracy(
            name="speaker_accuracy_real", threshold=0.)

        self.speaker_loss_converted_tracker = tf.metrics.Mean(name="speaker_loss_converted")
        self.speaker_accuracy_converted_tracker = tf.metrics.BinaryAccuracy(
            name="speaker_accuracy_converted", threshold=0.)

        self.feature_loss_tracker = tf.metrics.Mean(name="feature_loss")

        self.content_model = content_model
        self.content_model.trainable = False
        self.speaker_classification_model = speaker_classification_model

        self.gradient_clipping = gradient_clipping

        # OH NO IT'S HARD-CODED!!!
        self.logmel = LogMel(n_mels=128, n_fft=512, hop_len=128, sr=16000,
                             trainable=False, name="log_mel")
        self.logmel.build((None, None, 1))

    def train_step(self, data):
        source_audio, source_audio_length, _, _, _ = data[0]
        target_audio, target_audio_length, _, _, _ = data[1]

        source_spectrogram = self.logmel(source_audio)
        target_spectrogram = self.logmel(target_audio)

        # train conversion model
        # is this bad lol I dunno
        # the thing is that the speaker classifier variables count as trainable
        # variables for the conversion model... so we have to exclude them
        # here manually. I would suppose that there is a better way...
        conversion_variables = [variable for variable in self.trainable_variables
                                if not variable.name.startswith("CLASS")]
        with tf.GradientTape(watch_accessed_variables=False) as conversion_tape:
            for variable in conversion_variables:
                conversion_tape.watch(variable)

            conversion_spectrogram = self(source_spectrogram, training=True)

            # FIIRST WE SHALL train the content loss
            logits_source = self.content_model(source_spectrogram, training=False)
            logits_converted = self.content_model(conversion_spectrogram, training=False)

            # take into account mel transformation
            audio_length = tf.cast(
                tf.math.ceil(
                    (tf.cast(source_audio_length,
                             tf.float32) + 1) / self.content_model.hop_length),
                tf.int32)
            # take into account stride of the model
            audio_length = tf.cast(tf.math.ceil(audio_length / 2), tf.int32)
            mask = tf.sequence_mask(audio_length, dtype=tf.float32)[:, :, None]

            logits_squared_error = tf.math.squared_difference(logits_source,
                                                              logits_converted)

            masked_mse = tf.reduce_sum(mask * logits_squared_error) / tf.reduce_sum(mask)

            # NOW WHAT WE DO
            # train the generator part of GAN
            # for that we run the discriminator and try to confuse it (classify converted as REAL)

            # get the mask?? oh wey
            # we got two more strides of 2 in there so let's just do this LOL
            audio_lengths_source = [audio_length]*2
            audio_length_discriminator = tf.cast(tf.math.ceil(audio_length / 2), tf.int32)
            audio_lengths_source += [audio_length_discriminator]*2
            audio_length_discriminator = tf.cast(tf.math.ceil(audio_length_discriminator / 2), tf.int32)
            audio_lengths_source += [audio_length_discriminator]*2
            discriminator_mask = tf.sequence_mask(audio_length_discriminator, dtype=tf.float32)[:, :, None]

            discriminator_fake_output = self.speaker_classification_model(
                [conversion_spectrogram, discriminator_mask], training=False)
            discriminator_fake_logits = discriminator_fake_output[-1]
            fake_target_labels = tf.ones(tf.shape(discriminator_fake_logits)[0])[:, None]

            speaker_confusion = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=fake_target_labels, logits=discriminator_fake_logits))

            # we can add feature matching. for that we have to
            # - run fake audio through D (like above)
            # - also run real audio through D (like below...)
            # - for each layer compute differences between batch-averaged activations

            # also need mask for the target audio..............
            target_audio_length = tf.cast(
                tf.math.ceil(
                    (tf.cast(target_audio_length,
                             tf.float32) + 1) / self.content_model.hop_length),
                tf.int32)
            target_audio_length = tf.cast(
                tf.math.ceil(target_audio_length / 2), tf.int32)

            audio_lengths_target = [target_audio_length]*2
            target_audio_length = tf.cast(tf.math.ceil(target_audio_length / 2), tf.int32)
            audio_lengths_target += [target_audio_length]*2
            target_audio_length = tf.cast(tf.math.ceil(target_audio_length / 2), tf.int32)
            audio_lengths_target += [target_audio_length]*2
            discriminator_mask_target = tf.sequence_mask(target_audio_length,
                                                         dtype=tf.float32)[:, :, None]

            discriminator_real_features = self.speaker_classification_model(
                [target_spectrogram, discriminator_mask_target], training=False)[:-1]
            discriminator_fake_features = discriminator_fake_output[:-1]

            feature_loss = 0.
            for ind, (real_feature, fake_feature) in enumerate(
                    zip(discriminator_real_features, discriminator_fake_features)):
                masked_real = real_feature * tf.sequence_mask(audio_lengths_target[ind], dtype=tf.float32)[:, :, None]
                masked_fake = fake_feature * tf.sequence_mask(audio_lengths_source[ind], dtype=tf.float32)[:, :, None]
                masked_real_avg = tf.reduce_mean(masked_real, axis=[0, 1])
                masked_fake_avg = tf.reduce_mean(masked_fake, axis=[0, 1])
                feature_loss += tf.norm(masked_fake_avg - masked_real_avg)

            loss = masked_mse + speaker_confusion + feature_loss

        grads = conversion_tape.gradient(loss, conversion_variables)
        grads, global_norm = tf.clip_by_global_norm(
            grads, self.gradient_clipping)
        self.optimizer.apply_gradients(zip(grads, conversion_variables))

        # train discriminator
        # TODO maybe put in one batch
        # TODO spectral normalization in discriminator
        with tf.GradientTape() as classifier_tape:
            discriminator_fake_output = self.speaker_classification_model(
                [conversion_spectrogram, discriminator_mask], training=True)[-1]
            fake_target_labels = tf.zeros(tf.shape(discriminator_fake_output)[0])[:, None]
            speaker_loss_converted = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=fake_target_labels, logits=discriminator_fake_output))

            discriminator_real_output = self.speaker_classification_model(
                [target_spectrogram, discriminator_mask_target], training=True)[-1]
            real_target_labels = tf.ones(tf.shape(discriminator_real_output)[0])[:, None]
            speaker_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=0.9*real_target_labels, logits=discriminator_real_output))

            speaker_loss_total = 0.5 * (speaker_loss_converted + speaker_loss_real)

        speaker_grads = classifier_tape.gradient(
            speaker_loss_total, self.speaker_classification_model.trainable_variables)
        self.speaker_optimizer.apply_gradients(
            zip(speaker_grads,
                self.speaker_classification_model.trainable_variables))

        self.loss_tracker.update_state(masked_mse)
        self.speaker_confusion_tracker.update_state(speaker_confusion)

        self.speaker_loss_real_tracker.update_state(speaker_loss_real)
        self.speaker_accuracy_real_tracker(real_target_labels, discriminator_real_output)

        self.speaker_loss_converted_tracker.update_state(speaker_loss_converted)
        self.speaker_accuracy_converted_tracker(fake_target_labels, discriminator_fake_output)

        self.feature_loss_tracker.update_state(feature_loss)

        return {"reconstruction_loss": self.loss_tracker.result(),
                "speaker_confusion": self.speaker_confusion_tracker.result(),
                "speaker_loss_real": self.speaker_loss_real_tracker.result(),
                "speaker_accuracy_real": self.speaker_accuracy_real_tracker.result(),
                "speaker_loss_converted": self.speaker_loss_converted_tracker.result(),
                "speaker_accuracy_converted": self.speaker_accuracy_converted_tracker.result(),
                "feature_loss": self.feature_loss_tracker.result()}

    def test_step(self, data):
        # NOTE this only tests the content loss, not the speaker classification.
        # this is because speakers are disjoint between training and test sets
        source_audio, source_audio_length, _, _, _ = data[0]
        target_audio, target_audio_length, _, _, _ = data[1]
        audio_spectrogram = self.logmel(source_audio)

        with tf.GradientTape() as tape:
            reconstruction_spectrogram = self(audio_spectrogram, training=False)

            logits_target = self.content_model(audio_spectrogram, training=False)
            logits_recon = self.content_model(reconstruction_spectrogram, training=False)

            # take into account mel transformation
            audio_length = tf.cast(
                tf.math.ceil(
                    (tf.cast(source_audio_length,
                             tf.float32) + 1) / self.content_model.hop_length),
                tf.int32)
            # take into account stride of the model
            audio_length = tf.cast(tf.math.ceil(audio_length / 2), tf.int32)
            mask = tf.sequence_mask(audio_length, dtype=tf.float32)[:, :, None]

            logits_squared_error = tf.math.squared_difference(logits_target,
                                                              logits_recon)

            masked_mse = tf.reduce_sum(
                mask * logits_squared_error) / tf.reduce_sum(mask)
            loss = masked_mse

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker,
                self.speaker_confusion_tracker,
                self.speaker_loss_real_tracker,
                self.speaker_accuracy_real_tracker,
                self.speaker_loss_converted_tracker,
                self.speaker_accuracy_converted_tracker,
                self.feature_loss_tracker]


def build_voice_conversion_model(config: DictConfig) -> tf.keras.Model:
    # encoder-decoder model with 2d convolutions :shrug:
    logmel_input = tf.keras.Input((None, config.features.mel_freqs))
    x = logmel_input[..., None]  # add channel axis for 2d conv

    encoder_params = [(32, 3, 1), (64, 3, 1), (128, 3, 1), (256, 3, 1)]
    encoder_outputs = []
    for ind, (n_filters, width, stride) in enumerate(encoder_params):
        layer_string = "_encoder_" + str(ind)

        encoder_outputs.append(x)
        x = tfkl.Conv2D(n_filters, width, strides=2, padding="same",
                        use_bias=False, name="conv_stride" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn1" + layer_string, scale=False)(x)
        x = tfkl.LeakyReLU(alpha=.01, name="activation1" + layer_string)(x)

        x = tfkl.Conv2D(n_filters, width, strides=1, padding="same",
                        use_bias=False, name="conv2" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn2" + layer_string, scale=False)(x)
        x = tfkl.LeakyReLU(alpha=.01, name="activation2" + layer_string)(x)

    decoder_params = [(128, 3, 1), (64, 3, 1), (32, 3, 1)]
    for ind, (n_filters, width, stride) in enumerate(decoder_params):
        layer_string = "_decoder_" + str(ind)

        x = tfkl.Conv2D(n_filters, width, strides=stride, padding="same",
                        use_bias=False, name="conv1" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn1" + layer_string, scale=False)(x)
        x = tfkl.LeakyReLU(alpha=.01, name="activation1" + layer_string)(x)

        x = tfkl.UpSampling2D(2, name="upsample" + layer_string)(x)
        x = tfkl.Concatenate(name="concatenate" + layer_string)(
            [x, encoder_outputs[-(ind + 1)]])

        x = tfkl.Conv2D(n_filters, width, strides=stride, padding="same",
                        use_bias=False, name="conv2" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn2" + layer_string, scale=False)(x)
        x = tfkl.LeakyReLU(alpha=.01, name="activation2" + layer_string)(x)

    x = tfkl.UpSampling2D(2, name="upsample_decoder_final")(x)
    x = tfkl.Concatenate(name="concatenate_decoder_final")(
        [x, encoder_outputs[0]])
    reconstructed = tfkl.Conv2D(1, 1, name="conv_decoder_final")(x)
    reconstructed_no_channel = reconstructed[..., 0]

    wav2letter = build_w2l_model(28, config)
    wav2letter.load_weights(config.path.model + "_ref.h5")

    # XXX changed to accept logmel input directly
    # note we skip two layers -- 0 = input, 1 = logmel
    xwav = logmel_input
    for ind, layer in enumerate(wav2letter.layers[2:]):
        xwav = layer(xwav)

    wav2letter_logmel = tf.keras.Model(logmel_input, xwav)
    wav2letter_logmel.hop_length = wav2letter.hop_length

    speaker_classifier = build_discriminator(config)

    return ConversionModel(logmel_input, reconstructed_no_channel,
                           content_model=wav2letter_logmel,
                           speaker_classification_model=speaker_classifier,
                           gradient_clipping=config.training.gradient_clipping,
                           name="voice_conversion")


def build_discriminator(config):
    # XXX changed to accept logmel input directly
    logmel_input = tf.keras.Input((None, config.features.mel_freqs))
    a_mask = tf.keras.Input((None, 1))

    layer_params = [(256, 48, 2)] + [(256, 7, 1), (256, 7, 2)] * 2 + [(1024, 1, 1)]

    x = tfkl.LayerNormalization(name="CLASSinput_batchnorm", scale=True)(logmel_input)

    outputs = []

    for ind, (n_filters, width, stride) in enumerate(layer_params):
        layer_string = "_layer_" + str(ind)
        x = tfkl.Conv1D(n_filters, width, strides=stride, padding="same",
                        use_bias=False, name="CLASSconv" + layer_string)(x)
        x = tfkl.LayerNormalization(name="CLASSbn" + layer_string, scale=True)(x)
        x = tfkl.LeakyReLU(alpha=.01, name="CLASSactivation" + layer_string)(x)
        outputs.append(x)

    pre_pool_mask = a_mask * x
    pooled = tfkl.GlobalAveragePooling1D(name="CLASSglobal_pool")(pre_pool_mask)
    logits = tfkl.Dense(1, use_bias=True, name="CLASSlogits")(pooled)
    outputs.append(logits)

    return tf.keras.Model([logmel_input, a_mask], outputs, name="speaker_classifier")
