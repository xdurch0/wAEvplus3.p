import tensorflow as tf
from omegaconf import DictConfig

from .asr_model import build_w2l_model
from .utils.modeling import multiscale_spectrogram_loss

tfkl = tf.keras.layers


class ConversionModel(tf.keras.Model):
    def __init__(self, inputs, outputs,
                 content_model,
                 gradient_clipping, **kwargs):
        super().__init__(inputs, outputs, **kwargs)
        self.loss_tracker = tf.metrics.Mean(name="loss")
        self.content_model = content_model
        self.gradient_clipping = gradient_clipping

    def train_step(self, data):
        audio, audio_length, _, _ = data

        with tf.GradientTape() as tape:
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
            audio_length = tf.cast(audio_length / 2, tf.int32)
            mask = tf.sequence_mask(audio_length, dtype=tf.float32)[:, :, None]

            masked_mse = tf.reduce_sum(mask * logits_squared_error) / tf.reduce_sum(mask)
            loss = masked_mse

        grads = tape.gradient(loss, self.trainable_variables)
        grads, global_norm = tf.clip_by_global_norm(
            grads, self.gradient_clipping)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        audio, audio_length, _, _ = data

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
            audio_length = tf.cast(audio_length / 2, tf.int32)
            mask = tf.sequence_mask(audio_length, dtype=tf.float32)[:, :, None]

            masked_mse = tf.reduce_sum(
                mask * logits_squared_error) / tf.reduce_sum(mask)
            loss = masked_mse

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


def build_voice_conversion_model(config: DictConfig) -> tf.keras.Model:
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
    wav2letter.load_weights(config.path.model + "ref.h5")
    wav2letter.trainable = False

    return ConversionModel(wave_input, reconstructed,
                           content_model=wav2letter,
                           gradient_clipping=config.training.gradient_clipping,
                           name="voice_conversion")
