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
        self.speaker_loss_tracker = tf.metrics.Mean(name="speaker_loss")
        self.speaker_accuracy_tracker = tf.metrics.SparseCategoricalAccuracy(
            name="speaker_accuracy")
        self.topk_speaker_accuracy_tracker = tf.metrics.SparseTopKCategoricalAccuracy(
            5, name="top5_speaker_accuracy")
        self.speaker_confusion_tracker = tf.metrics.Mean(name="confusion_loss")

        self.content_model = content_model
        self.content_model.trainable = False
        self.speaker_classification_model = speaker_classification_model

        self.gradient_clipping = gradient_clipping

    def train_step(self, data):
        audio, audio_length, _, _, speaker_id = data

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

            reconstruction = self(audio, training=True)

            logits_target = self.content_model(audio, training=False)
            logits_recon = self.content_model(reconstruction, training=False)

            logits_squared_error = tf.math.squared_difference(logits_target,
                                                              logits_recon)
            # take into account mel transformation
            audio_length = tf.cast(
                tf.math.ceil(
                    (tf.cast(audio_length, tf.float32) + 1) / self.content_model.hop_length),
                tf.int32)
            # take into account stride of the model
            audio_length = tf.cast(tf.math.ceil(audio_length / 2), tf.int32)
            mask = tf.sequence_mask(audio_length, dtype=tf.float32)[:, :, None]

            masked_mse = tf.reduce_sum(mask * logits_squared_error) / tf.reduce_sum(mask)

            # confusion loss
            # dunno if this is good? basically we want to maximize entropy
            # but I'm scared of numerical issues with log!
            # so I optimize the loss between output distribution and a uniform
            # target distribution. might be the same mathematically??
            speaker_logits = self.speaker_classification_model(reconstruction,
                                                               training=True)
            uniform_targets = tf.ones_like(speaker_logits) / tf.cast(tf.shape(speaker_logits)[-1], tf.float32)
            speaker_confusion_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=uniform_targets, logits=speaker_logits))

            loss = masked_mse + speaker_confusion_loss

        grads = conversion_tape.gradient(loss, conversion_variables)
        grads, global_norm = tf.clip_by_global_norm(
            grads, self.gradient_clipping)
        self.optimizer.apply_gradients(zip(grads, conversion_variables))

        # train speaker classifier
        with tf.GradientTape() as classifier_tape:
            speaker_logits = self.speaker_classification_model(reconstruction,
                                                               training=True)
            speaker_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=speaker_id, logits=speaker_logits))

        speaker_grads = classifier_tape.gradient(
            speaker_loss, self.speaker_classification_model.trainable_variables)
        self.speaker_optimizer.apply_gradients(
            zip(speaker_grads,
                self.speaker_classification_model.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.speaker_loss_tracker.update_state(speaker_loss)
        self.speaker_accuracy_tracker(speaker_logits, speaker_id)
        self.topk_speaker_accuracy_tracker(speaker_logits, speaker_id)
        self.speaker_confusion_tracker.update_state(speaker_confusion_loss)

        return {"loss": self.loss_tracker.result(),
                "speaker_loss": self.speaker_loss_tracker.result(),
                "speaker_accuracy": self.speaker_accuracy_tracker.result(),
                "speaker_top5_accuracy": self.topk_speaker_accuracy_tracker.result(),
                "speaker_confusion": self.speaker_confusion_tracker.result()}

    def test_step(self, data):
        # NOTE this only tests the content loss, not the speaker classification.
        # this is because speakers are disjoint between training and test sets
        audio, audio_length, _, _, _ = data

        with tf.GradientTape() as tape:
            reconstruction = self(audio, training=False)

            logits_target = self.content_model(audio, training=False)
            logits_recon = self.content_model(reconstruction, training=False)

            logits_squared_error = tf.math.squared_difference(logits_target,
                                                              logits_recon)
            # take into account mel transformation
            audio_length = tf.cast(
                tf.math.ceil(
                    (tf.cast(audio_length,
                             tf.float32) + 1) / self.content_model.hop_length),
                tf.int32)
            # take into account stride of the model
            audio_length = tf.cast(tf.math.ceil(audio_length / 2), tf.int32)
            mask = tf.sequence_mask(audio_length, dtype=tf.float32)[:, :, None]

            masked_mse = tf.reduce_sum(
                mask * logits_squared_error) / tf.reduce_sum(mask)
            loss = masked_mse

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker,
                self.speaker_loss_tracker,
                self.speaker_accuracy_tracker,
                self.topk_speaker_accuracy_tracker,
                self.speaker_confusion_tracker]


def build_voice_conversion_model(config: DictConfig,
                                 n_speakers: int) -> tf.keras.Model:
    # encoder-decoder model with 1d convolutions :shrug:
    wave_input = tf.keras.Input((None, 1))

    encoder_params = [(32, 7, 1), (64, 7, 1), (128, 7, 1), (256, 7, 1)]
    x = wave_input
    encoder_outputs = []
    for ind, (n_filters, width, stride) in enumerate(encoder_params):
        layer_string = "_encoder_" + str(ind)

        encoder_outputs.append(x)
        x = tfkl.Conv1D(n_filters, width, strides=stride, padding="same",
                        use_bias=False, name="conv" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn" + layer_string, scale=False)(x)
        x = tfkl.ReLU(name="activation" + layer_string)(x)

        x = tfkl.MaxPool1D(4, padding="same", name="pool" + layer_string)(x)

    decoder_params = [(128, 7, 1), (64, 7, 1), (32, 7, 1)]
    for ind, (n_filters, width, stride) in enumerate(decoder_params):
        layer_string = "_decoder_" + str(ind)

        x = tfkl.UpSampling1D(4, name="upsample" + layer_string)(x)
        x = tfkl.Concatenate(name="concatenate" + layer_string)(
            [x, encoder_outputs[-(ind + 1)]])

        x = tfkl.Conv1D(n_filters, width, strides=stride, padding="same",
                        use_bias=False, name="conv" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn" + layer_string, scale=False)(x)
        x = tfkl.ReLU(name="activation" + layer_string)(x)

    x = tfkl.UpSampling1D(4, name="upsample_decoder_final")(x)
    x = tfkl.Concatenate(name="concatenate_decoder_final")(
        [x, encoder_outputs[0]])
    reconstructed = tf.nn.tanh(tfkl.Conv1D(1, 1, name="conv_decoder_final")(x))

    wav2letter = build_w2l_model(28, config)
    wav2letter.load_weights(config.path.model + "_ref.h5")

    speaker_classifier = build_speaker_classifier(config, n_speakers)

    return ConversionModel(wave_input, reconstructed,
                           content_model=wav2letter,
                           speaker_classification_model=speaker_classifier,
                           gradient_clipping=config.training.gradient_clipping,
                           name="voice_conversion")


def build_speaker_classifier(config, n_speakers):
    wave_input = tf.keras.Input((None, 1))

    layer_params = [(256, 48, 2)] + [(256, 7, 1), (256, 7, 2)] * 4 \
                   + [(2048, 7, 1), (2048, 1, 1)]

    x = LogMel(config.features.mel_freqs, config.features.window_size,
               config.features.hop_length, config.features.sample_rate,
               trainable=False, name="CLASSlog_mel")(wave_input)
    x = tfkl.BatchNormalization(name="CLASSinput_batchnorm", scale=False)(x)

    for ind, (n_filters, width, stride) in enumerate(layer_params):
        layer_string = "_layer_" + str(ind)
        x = tfkl.Conv1D(n_filters, width, strides=stride, padding="same",
                        use_bias=False, name="CLASSconv" + layer_string)(x)
        x = tfkl.BatchNormalization(name="CLASSbn" + layer_string, scale=False)(x)
        x = tfkl.ReLU(name="CLASSactivation" + layer_string)(x)

    pooled = tfkl.GlobalAveragePooling1D(name="CLASSglobal_pool")(x)
    logits = tfkl.Dense(n_speakers, use_bias=True, name="CLASSlogits")(pooled)

    return tf.keras.Model(wave_input, logits, name="speaker_classifier")
