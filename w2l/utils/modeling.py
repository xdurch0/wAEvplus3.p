import numpy as np
import tensorflow as tf


class CosineDecayWarmup(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            peak_learning_rate,
            warmup_steps,
            decay_steps,
            alpha=0.0,
            name=None):
        """Applies cosine decay to the learning rate.
        Args:
          peak_learning_rate: A scalar `float32` or `float64` Tensor or a
            Python number. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Number of steps to decay over.
          alpha: A scalar `float32` or `float64` Tensor or a Python number.
            Minimum learning rate value as a fraction of peak_learning_rate.
          name: String. Optional name of the operation.  Defaults to 'CosineDecay'.
        """
        super().__init__()

        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "CosineDecayWarmup"):
            if step <= self.warmup_steps:
                completed_fraction = tf.cast(step, tf.float32) / self.warmup_steps
                warmed_up = self.peak_learning_rate * completed_fraction
                capped = tf.maximum(warmed_up, 1e-10)
                tf.summary.scalar("decay_lr", capped,
                                  step=tf.cast(step, tf.int64))
                return capped
            else:
                peak_learning_rate = tf.convert_to_tensor(
                    self.peak_learning_rate, name="peak_learning_rate")
                dtype = peak_learning_rate.dtype
                decay_steps = tf.cast(self.decay_steps, dtype)

                global_step_recomp = tf.cast(step - self.warmup_steps, dtype)
                global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
                completed_fraction = global_step_recomp / decay_steps
                cosine_decayed = 0.5 * (1.0 + tf.cos(
                    tf.constant(np.pi) * completed_fraction))

                decayed = (1 - self.alpha) * cosine_decayed + self.alpha
                decayed_lr = tf.multiply(peak_learning_rate, decayed)
                tf.summary.scalar("decay_lr", decayed_lr,
                                  step=tf.cast(step, tf.int64))
                return decayed_lr

    def get_config(self):
        return {
            "peak_learning_rate": self.peak_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name}


def multiscale_spectrogram_loss(targets, outputs, audio_length):
    fft_sizes = [128, 256, 512, 1024, 2048]
    total_loss = 0

    for n_fft in fft_sizes:
        padded_targets = tf.pad(
            targets, ((0, 0), (n_fft // 2, n_fft // 2), (0, 0)),
            mode="reflect")
        padded_outputs = tf.pad(
            outputs, ((0, 0), (n_fft // 2, n_fft // 2), (0, 0)),
            mode="reflect")

        target_spectogram = tf.signal.stft(padded_targets[:, :, 0],
                                           n_fft, n_fft//4)
        output_spectrogram = tf.signal.stft(padded_outputs[:, :, 0],
                                            n_fft, n_fft//4)

        absolute_difference = tf.abs(target_spectogram - output_spectrogram)
        absolute_log_difference = tf.abs(tf.math.log(target_spectogram)
                                         - tf.math.log(output_spectrogram))

        spectrogram_lengths = tf.cast(
            tf.math.ceil((tf.cast(audio_length, tf.float32) + 1) / n_fft // 4), tf.int32)
        mask = tf.sequence_mask(spectrogram_lengths, dtype=tf.float32)

        contribution = tf.reduce_sum(mask * (absolute_difference + absolute_log_difference)) / tf.reduce_sum(mask)
        total_loss += contribution

    return total_loss
