import tensorflow as tf
from omegaconf import DictConfig

from asr_model import build_w2l_model
from input import w2l_dataset_npy
from utils.vocab import parse_vocab


def train_asr(config: DictConfig):
    print("Preparing dataset...")
    char_to_ind, ind_to_char = parse_vocab(config.path.vocab)
    train_dataset = w2l_dataset_npy(
        config,
        ["train-clean-100", "train-clean-360", "train-other-500"],
        True,
        char_to_ind,
        False)

    w2l = build_w2l_model(len(char_to_ind), config)
    optimizer = tf.optimizers.Adam(config.training.learning_rate)
    w2l.compile(optimizer=optimizer)
    history = w2l.fit(train_dataset,
                      validation_data=None,
                      epochs=config.training.epochs,
                      steps_per_epoch=100,
                      validation_steps=None,
                      callbacks=[])
