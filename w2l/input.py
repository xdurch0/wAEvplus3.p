import os
from typing import Iterable, Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig


def w2l_dataset_npy(config: DictConfig,
                    which_sets: Iterable[str],
                    vocab: Dict[str, int],
                    train: bool,
                    normalize: bool,
                    remap_ids: Optional[Dict] = None) -> Tuple[tf.data.Dataset, Dict]:
    """Builds a TF dataset for the preprocessed data.

    Parameters:
        config: hydra config object.
        which_sets: Contain all the subsets to be considered (e.g.
                    train-clean-360 etc.).
        vocab: Dictionary mapping characters to indices.
        train: Whether to shuffle and repeat data.
        normalize: If True, normalize each input array to mean 0, std 1.

    Returns:
        Dataset ready for consumption, and number of speakers.

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

    print("\tDoing random split...")
    np.random.seed(33)
    take_lines = np.random.rand(len(lines_split))
    if train:
        lines_split = [line for ind, line in enumerate(lines_split) if take_lines[ind] <= 0.8]
    else:
        lines_split = [line for ind, line in enumerate(lines_split) if take_lines[ind] > 0.8]

    print("\t{} entries remaining.".format(len(lines_split)))

    print("\tCreating the dataset...")
    ids, _, transcrs, subsets = zip(*lines_split)
    files = [os.path.join(config.path.array_dir, fid + ".npy") for fid in ids]
    if remap_ids is None:
        # process speaker ids.
        # we get the speaker id and remap to [0, 1, ..., n_speakers].
        speaker_ids = [full_id.split("-")[0] for full_id in ids]
        unique_speakers = sorted(set(speaker_ids))
        remap_ids = dict(zip(unique_speakers, range(len(unique_speakers))))

    def _to_arrays(fname, trans):
        return load_arrays_map_transcriptions(
            fname, trans, vocab, remap_ids, normalize)

    def gen():  # dummy to be able to use from_generator
        for file_name, transcr in zip(files, transcrs):
            # byte encoding is necessary in python 3, see TF 1.4 known issues
            yield file_name.encode("utf-8"), transcr.encode("utf-8")

    data = tf.data.Dataset.from_generator(
        gen, output_signature=(tf.TensorSpec((), dtype=tf.string),
                               tf.TensorSpec((), dtype=tf.string)))

    if train:
        # this basically shuffles the full dataset (~256k)
        data = data.shuffle(buffer_size=2**18).repeat()

    output_types = [tf.float32, tf.int32, tf.int32, tf.int32, tf.int32]
    data = data.map(
        lambda file_id, transcription: tuple(tf.numpy_function(
            _to_arrays, [file_id, transcription], output_types)),
        num_parallel_calls=tf.data.AUTOTUNE)
    # NOTE 1: padding value of 0 for element 1 and 3 is just a dummy (since
    #         sequence lengths are always scalar)
    # NOTE 2: changing padding value of -1 for element 2 requires changes
    # in the model as well!
    pad_shapes = ((-1, 1), (), (-1,), (), ())
    pad_values = (0., 0, -1, 0, 0)
    data = data.padded_batch(
        config.training.batch_size, padded_shapes=pad_shapes,
        padding_values=pad_values)
    # data = data.map(pack_inputs_in_dict, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.prefetch(tf.data.AUTOTUNE)

    return data, remap_ids


def load_arrays_map_transcriptions(file_name: bytes,
                                   trans: bytes,
                                   vocab: Dict[str, int],
                                   remap_ids: Dict[str, int],
                                   normalize: bool) -> Tuple[np.ndarray, int,
                                                             np.ndarray, int,
                                                             int]:
    """Mapping function to go from file names to numpy arrays.

    Goes from file_id, transcriptions to a tuple np_array, coded_transcriptions
    (integers).

    Parameters:
        file_name: Path built from ID taken from data csv, should match npy
                   file names. Expected to be utf-8 encoded as bytes.
        trans: Transcription. Also utf-8 bytes.
        vocab: Mapping of characters to integers.
        remap_ids: Mapping of speaker IDs in the csv to range from 0 to n.
        normalize: If True, normalize the array to peak amplitude 1.

    Returns:
        Tuple of 2D numpy array (seq_len x 1), scalar,
        1D array (label_len), scalar

    """
    array = np.load(file_name.decode("utf-8"))
    # temporary hack for the reconstruction thing: pad arrays so that they
    # are divisible by 256
    remainder = len(array) % 256
    if remainder:
        array = np.pad(array, ((0, 256 - remainder),))

    trans_mapped = np.array([vocab[ch] for ch in trans.decode("utf-8")],
                            dtype=np.int32)
    audio_length = np.int32(array.shape[-1])
    transcription_length = np.int32(len(trans_mapped))

    if normalize:
        array /= np.max(abs(array))

    _, file_id = os.path.split(file_name.decode("utf-8"))
    speaker_id = file_id.split("-")[0]
    remapped_id = np.int32(remap_ids[speaker_id])

    return_vals = (array[:, None].astype(np.float32), audio_length,
                   trans_mapped, transcription_length, remapped_id)

    return return_vals


def pack_inputs_in_dict(audio: tf.Tensor,
                        length: tf.Tensor,
                        trans: tf.Tensor,
                        trans_length: tf.Tensor) -> Tuple[Dict[str, tf.Tensor],
                                                          Dict[str, tf.Tensor]]:
    """For estimator interface (only allows one input -> pack into dict)."""
    return ({"audio": audio, "audio_length": length},
            {"transcription": trans, "transcription_length": trans_length})
