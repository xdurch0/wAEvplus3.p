import tensorflow as tf
from omegaconf import DictConfig

tfkl = tf.keras.layers


def build_conversion_model(config: DictConfig) -> tf.keras.Model:
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

    decoder_params = [(128, 7, 1), (64, 7, 1), (32, 7, 1)]
    for ind, (n_filters, width, stride) in enumerate(decoder_params):
        layer_string = "_decoder_" + str(ind)
        x = tfkl.Conv1D(n_filters, width, strides=stride, padding="same",
                        use_bias=False, name="conv" + layer_string)(x)
        x = tfkl.BatchNormalization(name="bn" + layer_string, scale=False)(x)
        x = tfkl.ReLU(name="activation" + layer_string)(x)

    reconstructed = tfkl.Conv1D(1, 1)(x)

    return tf.keras.Model(wave_input, reconstructed)