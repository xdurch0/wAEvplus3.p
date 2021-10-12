from typing import Dict, Union, Tuple

import librosa
import tensorflow as tf
from omegaconf import DictConfig

tfkl = tf.keras.layers


def dense_to_sparse(dense_tensor: tf.Tensor,
                    sparse_val: int = -1) -> tf.SparseTensor:
    """Inverse of tf.sparse_to_dense.
    Parameters:
        dense_tensor: The dense tensor. Duh.
        sparse_val: The value to "ignore": Occurrences of this value in the
                    dense tensor will not be represented in the sparse tensor.
                    NOTE: When/if later restoring this to a dense tensor, you
                    will probably want to choose this as the default value.
    Returns:
        SparseTensor equivalent to the dense input.
    """
    sparse_inds = tf.where(tf.not_equal(dense_tensor, sparse_val))
    sparse_vals = tf.gather_nd(dense_tensor, sparse_inds)
    dense_shape = tf.shape(dense_tensor, out_type=tf.int64)
    return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)


def inverse_softplus(x: Union[float, tf.Tensor]) -> tf.Tensor:
    return tf.math.log(tf.exp(x) - 1.)


class LogMel(tfkl.Layer):
    """Compute Mel spectrograms and apply logarithmic compression."""

    def __init__(self,
                 n_mels: int,
                 n_fft: int,
                 hop_len: int,
                 sr: int,
                 pad: bool = True,
                 compression: float = 1e-6,
                 **kwargs):
        """Prepare variables for conversion.
        Parameters:
            n_mels: Number of mel frequency bands.
            n_fft: Size of FFT window.
            hop_len: Hop size between FFT applications.
            sr: Sampling rate of audio.
            pad: Whether to pad first/last FFT windows. This means frames will
                 be "centered" around time instead of "left-aligned".
            compression: Additive offset for log compression.
            kwargs: Arguments for tfkl.Layer.
        """
        super().__init__(**kwargs)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.pad = pad

        to_mel = librosa.filters.mel(sr, n_fft, n_mels=n_mels).T

        self.mel_matrix = tf.Variable(initial_value=to_mel,
                                      trainable=False,
                                      dtype=tf.float32,
                                      name=self.name + "_weights")
        self.compression = tf.Variable(
            initial_value=tf.ones(n_mels) * inverse_softplus(compression),
            trainable=self.trainable,
            dtype=tf.float32,
            name=self.name + "_compression")

    def call(self,
             inputs: tf.Tensor,
             **kwargs) -> tf.Tensor:
        """Apply the layer.
        Parameters:
            inputs: Audio. Note that we assume a channel axis (size 1) even
                    though it is not used. Compatibility reasons.
            kwargs: Other arguments to Layer.call; ignored.
        Returns:
            Log-power mel spectrogram.
        """
        if self.pad:
            inputs = tf.pad(
                inputs, ((0, 0), (self.n_fft // 2, self.n_fft // 2), (0, 0)),
                mode="reflect")

        spectros = tf.signal.stft(inputs[:, :, 0], self.n_fft, self.hop_len)
        power = tf.abs(spectros) ** 2

        mel = tf.matmul(power, self.mel_matrix)
        logmel = tf.math.log(mel + tf.nn.softplus(self.compression))

        return logmel


class W2L(tf.keras.Model):
    def __init__(self,
                 inputs: tf.Tensor,
                 outputs: tf.Tensor):
        super().__init__(inputs, outputs)
        self.loss_tracker = tf.metrics.Mean(name="loss")

    def train_step(self,
                   data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor])\
            -> Dict[str, tf.Tensor]:
        audio, audio_length, transcriptions, transcription_length = data

        # take into account mel transformation
        # TODO HORRIBLE MAGIC NUMBERS OH GOD
        audio_length = tf.cast(tf.math.ceil((tf.cast(audio_length, tf.float32) + 1) / 128), tf.int32)
        # take into account stride of the model
        audio_length = tf.cast(audio_length / 2, tf.int32)

        transcriptions_sparse = dense_to_sparse(transcriptions)

        with tf.GradientTape() as tape:
            logits = self(audio, training=True)
            # after this we need logits in shape time x batch_size x vocab_size
            logits_time_major = tf.transpose(logits, [1, 0, 2])

            # note this is the "CPU version" which may be slower, but earlier
            # attempts at using the GPU version resulted in catastrophe...
            ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
                labels=transcriptions_sparse,
                logits=logits_time_major,
                label_length=None,
                logit_length=audio_length,
                logits_time_major=True,
                blank_index=0))

        grads = tape.gradient(ctc_loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(ctc_loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self,
                  data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor])\
            -> Dict[str, tf.Tensor]:
        audio, audio_length, transcriptions, transcription_length = data

        # take into account mel transformation
        # TODO HORRIBLE MAGIC NUMBERS OH GOD
        audio_length = tf.cast(
            tf.math.ceil((tf.cast(audio_length, tf.float32) + 1) / 128), tf.int32)
        # take into account stride of the model
        audio_length = tf.cast(audio_length / 2, tf.int32)

        transcriptions_sparse = dense_to_sparse(transcriptions)

        logits = self(audio, training=True)
        # after this we need logits in shape time x batch_size x vocab_size
        logits_time_major = tf.transpose(logits, [1, 0, 2])

        ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
            labels=transcriptions_sparse,
            logits=logits_time_major,
            label_length=None,
            logit_length=audio_length,
            logits_time_major=True,
            blank_index=0))

        self.loss_tracker.update_state(ctc_loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


def build_w2l_model(vocab_size: int,
                    config: DictConfig) -> W2L:
    wave_input = tf.keras.Input((None, 1))

    layer_params = [(256, 48, 2)] + [(256, 7, 1)]*8 + [(2048, 32, 1),
                                                       (2048, 1, 1)]

    x = LogMel(config.features.mel_freqs, config.features.window_size,
               config.features.hop_length, config.features.sample_rate,
               trainable=False)(wave_input)
    x = tfkl.BatchNormalization()(x)
    for n_filters, width, stride in layer_params:
        x = tfkl.Conv1D(n_filters, width, strides=stride, padding="same")(x)
        x = tfkl.BatchNormalization()(x)
        x = tfkl.ELU()(x)
    logits = tfkl.Conv1D(vocab_size + 1, 1)(x)

    w2l = W2L(wave_input, logits)

    return w2l
