import librosa
import tensorflow as tf
tfkl = tf.keras.layers


def inverse_softplus(x):
    return tf.math.log(tf.exp(x) - 1.)


class LogMel(tfkl.Layer):
    """Compute Mel spectrograms and apply logarithmic compression."""

    def __init__(self,
                 n_mels: int,
                 n_fft: int,
                 hop_len: int,
                 sr: int,
                 pad: bool = True,
                 compression: float = 1e-8,
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

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
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
    def train_step(self, data: tuple) -> dict:
        audio, transcriptions = data
        with tf.GradientTape() as tape:
            logits = self(audio, training=True)
            # after this we need logits in shape time x batch_size x vocab_size
            if self.cf:  # bs x v x t -> t x bs x v
                logits_tm = tf.transpose(logits, [2, 0, 1],
                                         name="logits_time_major")
            else:  # channels last: bs x t x v -> t x bs x v
                logits_tm = tf.transpose(logits, [1, 0, 2],
                                         name="logits_time_major")

            audio_length = tf.cast(audio_length / 2, tf.int32)

            if False:  # on_gpu:  # this seems to be slow so we don't use it
                ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
                    labels=transcrs, logits=logits_tm,
                    label_length=transcr_length,
                    logit_length=audio_length, logits_time_major=True,
                    blank_index=0), name="avg_loss")
            else:
                transcrs_sparse = dense_to_sparse(transcrs, sparse_val=-1)
                ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(
                    labels=transcrs_sparse, logits=logits_tm, label_length=None,
                    logit_length=audio_length, logits_time_major=True,
                    blank_index=0), name="avg_loss")

            if self.regularizer_coeff:
                avg_reg_loss = tf.math.add_n(self.model.losses) / len(
                    self.model.losses)
                loss = ctc_loss + self.regularizer_coeff * avg_reg_loss
            else:
                loss = ctc_loss
                avg_reg_loss = 0

        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # probably has to go into train_full...
        # self.annealer.update_history(loss)

        return ctc_loss, avg_reg_loss


def build_w2l_model(vocab_size, config):
    wave_input = tf.keras.Input((None, 1))

    layer_params = [(256, 48, 2)] + [(256, 7, 1)]*8 + [(2048, 32, 1), (2048, 1, 1)]

    x = LogMel(config.features.mel_freqs, config.features.window_size,
               config.features.hop_length, config.features.sample_rate,
               trainable=False)(wave_input)
    for n_filters, width, stride in layer_params:
        x = tfkl.Conv1D(n_filters, width, stride=stride, padding="same")(x)
        x = tfkl.BatchNormalization(x)
        x = tfkl.ReLU()(x)
    logits = tfkl.Conv1D(vocab_size + 1, 1)(x)

    w2l = tf.keras.Model(wave_input, logits)

    return w2l
