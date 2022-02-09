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

        # OH NO IT'S HARD-CODED!!!
        self.logmel = LogMel(n_mels=128, n_fft=512, hop_len=128, sr=16000,
                             trainable=False, name="log_mel")
        self.logmel.build((None, None, 1))

    def train_step(self, data):
        audio, audio_length, _, _, speaker_id = data

        audio_spectrogram = self.logmel(audio)

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

            reconstruction_spectrogram = self(audio_spectrogram, training=True)

            logits_target = self.content_model(audio_spectrogram, training=False)
            logits_recon = self.content_model(reconstruction_spectrogram, training=False)

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
            # DIFFERENT IDEA could be:
            # cross-entropy with targets = softmax(logits) is just entropy
            # maximizing entropy -> uniform distribution.
            # so use negative entropy as loss!
            # to do this, use speaker_probabilities as label
            # along with non-sparse cross-entropy
            speaker_logits = self.speaker_classification_model(reconstruction_spectrogram,
                                                               training=False)
            # speaker_probabilities = tf.nn.softmax(speaker_logits)
            speaker_confusion = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=speaker_id, logits=speaker_logits))

            loss = masked_mse - speaker_confusion

        grads = conversion_tape.gradient(loss, conversion_variables)
        grads, global_norm = tf.clip_by_global_norm(
            grads, self.gradient_clipping)
        self.optimizer.apply_gradients(zip(grads, conversion_variables))

        # train speaker classifier
        with tf.GradientTape() as classifier_tape:
            speaker_logits = self.speaker_classification_model(reconstruction_spectrogram,
                                                               training=True)
            speaker_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=speaker_id, logits=speaker_logits))

        speaker_grads = classifier_tape.gradient(
            speaker_loss, self.speaker_classification_model.trainable_variables)
        self.speaker_optimizer.apply_gradients(
            zip(speaker_grads,
                self.speaker_classification_model.trainable_variables))

        self.loss_tracker.update_state(masked_mse)
        self.speaker_loss_tracker.update_state(speaker_loss)
        self.speaker_accuracy_tracker(speaker_id, speaker_logits)
        self.topk_speaker_accuracy_tracker(speaker_id, speaker_logits)
        self.speaker_confusion_tracker.update_state(speaker_confusion)

        return {"reconstruction_loss": self.loss_tracker.result(),
                "speaker_loss": self.speaker_loss_tracker.result(),
                "speaker_accuracy": self.speaker_accuracy_tracker.result(),
                "speaker_top5_accuracy": self.topk_speaker_accuracy_tracker.result(),
                "speaker_confusion": self.speaker_confusion_tracker.result()}

    def test_step(self, data):
        # NOTE this only tests the content loss, not the speaker classification.
        # this is because speakers are disjoint between training and test sets
        audio, audio_length, _, _, _ = data
        audio_spectrogram = self.logmel(audio)

        with tf.GradientTape() as tape:
            reconstruction_spectrogram = self(audio_spectrogram, training=False)

            logits_target = self.content_model(audio_spectrogram, training=False)
            logits_recon = self.content_model(reconstruction_spectrogram, training=False)

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
    # encoder-decoder model with 2d convolutions :shrug:
    logmel_input = tf.keras.Input((None, config.features.mel_freqs))
    x = logmel_input[..., None]  # add channel axis for 2d conv

    encoder_params = [(16, 3, 1), (32, 3, 1), (64, 3, 1), (128, 3, 1)]
    encoder_outputs = []
    for ind, (n_filters, width, stride) in enumerate(encoder_params):
        layer_string = "_encoder_" + str(ind)

        encoder_outputs.append(x)
        x = tfkl.Conv2D(n_filters, width, strides=2, padding="same",
                        use_bias=False, name="conv_stride" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn1" + layer_string, scale=False)(x)
        x = tfkl.ReLU(name="activation1" + layer_string)(x)

        x = tfkl.Conv2D(n_filters, width, strides=1, padding="same",
                        use_bias=False, name="conv2" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn2" + layer_string, scale=False)(x)
        x = tfkl.ReLU(name="activation2" + layer_string)(x)

    decoder_params = [(64, 3, 1), (32, 3, 1), (16, 3, 1)]
    for ind, (n_filters, width, stride) in enumerate(decoder_params):
        layer_string = "_decoder_" + str(ind)

        x = tfkl.Conv2D(n_filters, width, strides=stride, padding="same",
                        use_bias=False, name="conv1" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn1" + layer_string, scale=False)(x)
        x = tfkl.ReLU(name="activation1" + layer_string)(x)

        x = tfkl.UpSampling2D(2, name="upsample" + layer_string)(x)
        x = tfkl.Concatenate(name="concatenate" + layer_string)(
            [x, encoder_outputs[-(ind + 1)]])

        x = tfkl.Conv2D(n_filters, width, strides=stride, padding="same",
                        use_bias=False, name="conv2" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn2" + layer_string, scale=False)(x)
        x = tfkl.ReLU(name="activation2" + layer_string)(x)

    x = tfkl.UpSampling2D(2, name="upsample_decoder_final")(x)
    x = tfkl.Concatenate(name="concatenate_decoder_final")(
        [x, encoder_outputs[0]])
    reconstructed = tfkl.Conv2D(1, 1, name="conv_decoder_final")(x)
    reconstructed_no_channel = reconstructed[..., 0]

    wav2letter = build_w2l_model(28, config)
    wav2letter.load_weights(config.path.model + "_ref.h5")

    # XXX changed to accept logmel input directly
    xwav = logmel_input
    for layer in wav2letter.layers[2:]:
        xwav = layer(xwav)
    wav2letter_logmel = tf.keras.Model(logmel_input, xwav)
    wav2letter_logmel.hop_length = wav2letter.hop_length

    speaker_classifier = build_speaker_classifier(config, n_speakers)

    return ConversionModel(logmel_input, reconstructed_no_channel,
                           content_model=wav2letter_logmel,
                           speaker_classification_model=speaker_classifier,
                           gradient_clipping=config.training.gradient_clipping,
                           name="voice_conversion")


def build_speaker_classifier(config, n_speakers):
    # XXX changed to accept logmel input directly
    logmel_input = tf.keras.Input((None, config.features.mel_freqs))

    layer_params = [(256, 48, 2)] + [(256, 7, 1), (256, 7, 2)] * 2 + [(1024, 1, 1)]

    x = tfkl.BatchNormalization(name="CLASSinput_batchnorm", scale=False)(logmel_input)

    for ind, (n_filters, width, stride) in enumerate(layer_params):
        layer_string = "_layer_" + str(ind)
        x = tfkl.Conv1D(n_filters, width, strides=stride, padding="same",
                        use_bias=False, name="CLASSconv" + layer_string)(x)
        x = tfkl.BatchNormalization(name="CLASSbn" + layer_string, scale=False)(x)
        x = tfkl.ReLU(name="CLASSactivation" + layer_string)(x)

    pooled = tfkl.GlobalAveragePooling1D(name="CLASSglobal_pool")(x)
    logits = tfkl.Dense(n_speakers, use_bias=True, name="CLASSlogits")(pooled)

    return tf.keras.Model(logmel_input, logits, name="speaker_classifier")
