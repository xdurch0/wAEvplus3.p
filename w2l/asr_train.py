import tensorflow as tf
from omegaconf import DictConfig

from .asr_model import build_w2l_model
from .input import w2l_dataset_npy
from .utils.vocab import parse_vocab


def train_asr(config: DictConfig):
    print("Preparing dataset...")
    char_to_ind, ind_to_char = parse_vocab(config.path.vocab)
    train_dataset = w2l_dataset_npy(
        config,
        ["train-clean-100"],
        True,
        char_to_ind,
        False)

    val_dataset = w2l_dataset_npy(config, ["dev-clean"], False, char_to_ind,
                                  False)

    w2l = build_w2l_model(len(char_to_ind), config)
    optimizer = tf.optimizers.Adam(config.training.learning_rate)
    w2l.compile(optimizer=optimizer)

    callback_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        patience=100, verbose=1)
    callback_tensorboard = tf.keras.callbacks.TensorBoard(
        histogram_freq=10, write_steps_per_second=True, update_freq="epoch")
    callback_stop = tf.keras.callbacks.EarlyStopping(patience=300)
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        config.path.model, save_best_only=True)
    callbacks = [callback_plateau, callback_tensorboard, callback_stop,
                 callback_checkpoint]

    history = w2l.fit(train_dataset,
                      validation_data=val_dataset,
                      epochs=config.training.epochs,
                      steps_per_epoch=100,
                      validation_steps=None,
                      callbacks=callbacks)
