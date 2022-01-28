import os
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa
from omegaconf import DictConfig

from .conversion_model import build_voice_conversion_model
from .input import w2l_dataset_npy
from .utils.modeling import CosineDecayWarmup
from .utils.vocab import parse_vocab


def train_conversion(config: DictConfig):
    print("Preparing dataset...")
    char_to_ind, ind_to_char = parse_vocab(config.path.vocab)

    if config.training.subsets == "all":
        train_subsets = ["train-clean-100", "train-clean-360",
                         "train-other-500"]
        val_subsets = ["dev-clean", "dev-other"]
    elif config.training.subsets == "small":
        train_subsets = ["train-clean-100", "train-clean-360"]
        val_subsets = ["dev-clean"]
        train_subsets = train_subsets + val_subsets
        val_subsets = train_subsets
    else:
        raise ValueError("Invalid subsets specified.")

    train_dataset, speaker_ids = w2l_dataset_npy(
        config,
        train_subsets,
        char_to_ind,
        train=True,
        normalize=False)

    val_dataset, _ = w2l_dataset_npy(
        config,
        val_subsets,
        char_to_ind,
        train=False,
        normalize=False,
        remap_ids=speaker_ids)

    n_speakers = len(speaker_ids)

    conversion_model = build_voice_conversion_model(
        config, n_speakers=n_speakers)

    lr_schedule = CosineDecayWarmup(
        peak_learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_epochs*config.training.steps_per_epoch,
        decay_steps=(config.training.epochs - config.training.warmup_epochs)*config.training.steps_per_epoch)
    optimizer = tfa.optimizers.AdamW(
        weight_decay=config.training.weight_decay,
        learning_rate=lr_schedule)

    conversion_model.compile(optimizer=optimizer, run_eagerly=True)
    # TODO kinda bad to put that herew
    conversion_model.speaker_optimizer = tfa.optimizers.AdamW(
        weight_decay=config.training.weight_decay,
        learning_rate=lr_schedule)

    time_string = str(datetime.now())
    tb_logdir = os.path.join(config.path.logs + "_speakerinfo",
                             "run_" + time_string)
    model_path = config.path.model + "_speakerinfo_" + time_string + ".h5"
    callback_tensorboard = tf.keras.callbacks.TensorBoard(
        histogram_freq=1, write_steps_per_second=True, update_freq=100,
        log_dir=tb_logdir, profile_batch=0)
    callback_stop = tf.keras.callbacks.EarlyStopping(
        patience=20000 // config.training.steps_per_epoch, verbose=1,
        monitor="val_speaker_loss", mode="min")
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, save_best_only=True, save_weights_only=True,
        verbose=1)
    callbacks = [callback_tensorboard, callback_stop, callback_checkpoint,
                 tf.keras.callbacks.TerminateOnNaN()]

    print(conversion_model.summary())
    print(conversion_model.speaker_classification_model.summary())

    history = conversion_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.training.epochs,
        steps_per_epoch=config.training.steps_per_epoch,
        validation_steps=None,
        callbacks=callbacks)
