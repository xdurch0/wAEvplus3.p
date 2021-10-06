import os
from typing import Iterable, Dict

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig


def w2l_input_fn_npy(config: DictConfig,
                     which_sets: Iterable[str],
                     train: bool,
                     vocab: Dict[str, int],
                     normalize) -> tf.data.Dataset:
    """Builds a TF dataset for the preprocessed data.

    Parameters:
        config: hydra config object.
        which_sets: Contain all the subsets to be considered (e.g.
                    train-clean-360 etc.).
        train: Whether to shuffle and repeat data.
        vocab: Dictionary mapping characters to indices.
        normalize: Bool; if True, normalize each input array to mean 0, std 1.

    Returns:
        Dataset ready for consumption.

    """
    print("Building dataset for {} set using file {}...".format(
        which_sets, config.path.csv))
    # first read the csv and keep the useful stuff
    with open(config.path.csv, mode="r") as corpus:
        lines_split = [line.strip().split(",") for line in corpus]
    print("\t{} entries found.".format(len(lines_split)))

    if which_sets:
        print("\tFiltering requested subset...")
        lines_split = [line for line in lines_split if line[3] in which_sets]
    if not lines_split:
        raise ValueError("Filtering resulted in size-0 dataset! Maybe you "
                         "specified an invalid subset? You supplied "
                         "'{}'.".format(which_sets))
    print("\t{} entries remaining.".format(len(lines_split)))

    print("\tCreating the dataset...")
    ids, _, transcrs, subsets = zip(*lines_split)
    files = [os.path.join(config.path.array_dir, fid + ".npy") for fid in ids]

    def _to_arrays(fname, trans):
        return _pyfunc_load_arrays_map_transcriptions(
            fname, trans, vocab, normalize)

    def gen():  # dummy to be able to use from_generator
        for file_name, transcr in zip(files, transcrs):
            # byte encoding is necessary in python 3, see TF 1.4 known issues
            yield file_name.encode("utf-8"), transcr.encode("utf-8")

    data = tf.data.Dataset.from_generator(
        gen, output_signature=(tf.TensorSpec((None,), dtype=tf.string),
                               tf.TensorSpec((None,), dtype=tf.string)))

    if train:
        # this basically shuffles the full dataset (~256k)
        data = data.shuffle(buffer_size=2**18).repeat()

    output_types = [tf.float32, tf.int32, tf.int32, tf.int32]
    data = data.map(
        lambda file_id, transcription: tuple(tf.numpy_function(
            _to_arrays, [file_id, transcription], output_types)),
        num_parallel_calls=tf.data.AUTOTUNE)
    # NOTE 1: padding value of 0 for element 1 and 3 is just a dummy (since
    #         sequence lengths are always scalar)
    # NOTE 2: changing padding value of -1 for element 2 requires changes
    # in the model as well!
    pad_shapes = ((-1, 1), (), (-1,), ())
    pad_values = (0., 0, -1, 0)
    data = data.padded_batch(
        config.training.batch_size, padded_shapes=pad_shapes,
        padding_values=pad_values)
    map_fn = pack_inputs_in_dict
    data = data.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.prefetch(tf.data.AUTOTUNE)

    return data


def _pyfunc_load_arrays_map_transcriptions(file_name, trans, vocab, normalize):
    """Mapping function to go from file names to numpy arrays.

    Goes from file_id, transcriptions to a tuple np_array, coded_transcriptions
    (integers).
    NOTE: Files are assumed to be stored channels_first. If this is not the
          case, this will cause trouble down the line!!
    Parameters:
        file_name: Path built from ID taken from data csv, should match npy
                   file names. Expected to be utf-8 encoded as bytes.
        trans: Transcription. Also utf-8 bytes.
        vocab: Dictionary mapping characters to integers.
        normalize: Bool; if True, normalize the array to mean 0, std 1.

    Returns:
        Tuple of 2D numpy array (seq_len x 1), scalar,
        1D array (label_len), scalar

    """
    array = np.load(file_name.decode("utf-8"))
    trans_mapped = np.array([vocab[ch] for ch in trans.decode("utf-8")],
                            dtype=np.int32)
    audio_length = np.int32(array.shape[-1])
    transcription_length = np.int32(len(trans_mapped))

    if normalize:
        array = (array - np.mean(array)) / np.std(array)

    return_vals = (array[:, None].astype(np.float32), audio_length,
                   trans_mapped, transcription_length)

    return return_vals


def pack_inputs_in_dict(audio, length, trans, trans_length):
    """For estimator interface (only allows one input -> pack into dict)."""
    return ({"audio": audio, "length": length},
            {"transcription": trans, "length": trans_length})
