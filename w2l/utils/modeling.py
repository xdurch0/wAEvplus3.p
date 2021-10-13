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
                return tf.maximum(warmed_up, 1e-8)
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
                return tf.multiply(peak_learning_rate, decayed)

    def get_config(self):
        return {
            "peak_learning_rate": self.peak_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name
        }
