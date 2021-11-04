import tensorflow as tf
from omegaconf import DictConfig

from .utils.modeling import multiscale_spectrogram_loss

tfkl = tf.keras.layers


class ConversionModel(tf.keras.Model):
    def __init__(self, inputs, outputs, gradient_clipping, **kwargs):
        super().__init__(inputs, outputs, **kwargs)
        self.loss_tracker = tf.metrics.Mean(name="loss")
        self.gradient_clipping = gradient_clipping

    def train_step(self, data):
        audio, audio_length, _, _ = data

        with tf.GradientTape() as tape:
            reconstruction = self(audio, training=True)

            loss = multiscale_spectrogram_loss(audio, reconstruction,
                                               audio_length)

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

            loss = multiscale_spectrogram_loss(audio, reconstruction,
                                               audio_length)

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
    for ind, (n_filters, width, stride) in enumerate(encoder_params):
        layer_string = "_encoder_" + str(ind)
        x = tfkl.Conv1D(n_filters, width, strides=stride, padding="same",
                        use_bias=False, name="conv" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn" + layer_string, scale=False)(x)
        x = tfkl.ReLU(name="activation" + layer_string)(x)
        x = tfkl.MaxPool1D(4, padding="same", name="pool" + layer_string)(x)

    decoder_params = [(128, 7, 1), (64, 7, 1), (32, 7, 1)]
    for ind, (n_filters, width, stride) in enumerate(decoder_params):
        layer_string = "_decoder_" + str(ind)
        x = tfkl.UpSampling1D(4)(x)
        x = tfkl.Conv1D(n_filters, width, strides=stride, padding="same",
                        use_bias=False, name="conv" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn" + layer_string, scale=False)(x)
        x = tfkl.ReLU(name="activation" + layer_string)(x)
    x = tfkl.UpSampling1D(4)(x)
    reconstructed = tfkl.Conv1D(1, 1)(x)

    return ConversionModel(wave_input, reconstructed,
                           gradient_clipping=config.training.gradient_clipping,
                           name="voice_conversion")
