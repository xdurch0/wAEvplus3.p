import tensorflow as tf
import tensorflow_addons as tfa
from omegaconf import DictConfig

from .asr_model import build_w2l_model
from .input import w2l_dataset_npy
from .utils.modeling import CosineDecayWarmup
from .utils.vocab import parse_vocab


def train_asr(config: DictConfig):
    print("Preparing dataset...")
    char_to_ind, ind_to_char = parse_vocab(config.path.vocab)
    train_dataset = w2l_dataset_npy(
        config,
        ["train-clean-100", "train-clean-360", "train-other-500"],
        char_to_ind,
        train=True,
        normalize=False)

    val_dataset = w2l_dataset_npy(config, ["dev-clean", "dev-other"],
                                  char_to_ind, train=False, normalize=False)

    w2l = build_w2l_model(len(char_to_ind), config)

    lr_schedule = CosineDecayWarmup(
        peak_learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_epochs*config.training.steps_per_epoch,
        decay_steps=(config.training.epochs - config.training.warmup_epochs)*config.training.steps_per_epoch)
    optimizer = tfa.optimizers.AdamW(
        weight_decay=config.training.weight_decay,
        learning_rate=lr_schedule)

    w2l.compile(optimizer=optimizer, run_eagerly=True)

    callback_tensorboard = tf.keras.callbacks.TensorBoard(
        histogram_freq=1, write_steps_per_second=True, update_freq=100,
        log_dir=config.path.logs, profile_batch=0)
    callback_stop = tf.keras.callbacks.EarlyStopping(patience=4, verbose=1)
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        config.path.model, save_best_only=True)
    callbacks = [callback_tensorboard, callback_stop, callback_checkpoint]

    history = w2l.fit(train_dataset,
                      validation_data=val_dataset,
                      epochs=config.training.epochs,
                      steps_per_epoch=config.training.steps_per_epoch,
                      validation_steps=None,
                      callbacks=callbacks)
