import os
from typing import Optional

import hydra
import librosa
import numpy as np
from omegaconf import DictConfig

from w2l.utils.vocab import make_vocab
w

def make_corpus_csv(base_path: str,
                    out_path: str):
    """Create a csv containing corpus info from a given directory.

    This will scan all subfolders recursively. For any folder that has files, it
    is assumed that these files are:
    - n audio files containing speech, each with a unique ID.
    - 1 text file with n lines, containing in each line a file id and the
      corresponding transcription, separated by a space

    NOTE the transcription file should be alphanumerically last (so it will be
    sorted at the end). TODO change this so we just look for a .txt file

    NOTE it is assumed that the top-level folder is NOT a data directory, i.e.
    it will be skipped! This means you can use this folder to put stuff like
    readmes etc.
    Also, the level below top-level is assumed to represent different "subsets"
    of data. Later you can choose to only pick (or exclude) certain subsets. For
    example you could have a "train" folder and a "test" folder, or different
    subdivisions of these.

    The csv will contain lines as follows (1 line per file as mentioned above):
        id, filepath, transcription, set
            id: File id.
            filepath: Relative path to the original audio file from the base
                      directory.
            transcription: The text.
            set: Which subset this came from (or is to be used for), e.g. for
            LibriSpeech this could be train-clean-360 or dev-other.

    Parameters:
        base_path: Path to corpus, e.g. /data/LibriSpeech.
        out_path: Path you want the corpus csv to go to.

    """
    print("Creating {} from {}...".format(out_path, base_path))

    with open(out_path, mode="w") as corpus_csv:
        for subset in os.listdir(base_path):
            joined = os.path.join(base_path, subset)
            if not os.path.isdir(joined):
                continue

            print("\tProcessing {}...".format(subset))
            corpus_walker = os.walk(joined)
            for path, _, files in corpus_walker:
                if not files:  # not a data directory
                    continue

                files = sorted(files)  # puts transcriptions at the end
                transcrs = open(os.path.join(path, files[-1])).readlines()
                # the below line removes the IDs and puts the transcriptions
                # back together
                transcrs = [" ".join(t.strip().split()[1:]).lower()
                            for t in transcrs]
                if len(files[:-1]) != len(transcrs):
                    raise ValueError(
                        "Discrepancy in {}: {} audio files found,"
                        " but {} transcriptions (should be the "
                        "same).".format(path, len(files[:-1]), len(transcrs)))

                for f, t in zip(files[:-1], transcrs):
                    # file ID, relative path, transcription, corpus
                    fid = f.split(".")[0]
                    fpath = os.path.join(subset, path, f)
                    corpus_csv.write(",".join([fid, fpath, t, subset]) + "\n")


def preprocess_audio(csv_path: str,
                     corpus_path: str,
                     array_dir: str,
                     resample_rate: Optional[int] = None):
    """Preprocess many audio files with requested parameters.

    Parameters:
        csv_path: Path to corpus csv.
        corpus_path: Path to corpus.
        array_dir: Path to directory where all the processed arrays should be
                   stored in.
        resample_rate: Hz to resample data to. If not given, no resampling
                       is performed and any sample rate != 16000 leads to a
                       crash.

    """
    os.mkdir(array_dir)
    with open(csv_path) as corpus_csv:
        for n, line in enumerate(corpus_csv, start=1):
            file_id, file_path, _, subset = line.strip().split(",")
            path = os.path.join(corpus_path, file_path)
            audio, sr = librosa.load(path, sr=resample_rate)
            if sr != 16000:
                raise ValueError("Sampling rate != 16000 found in "
                                 "{}!".format(path))

            np.save(os.path.join(array_dir, file_id + ".npy"),
                    audio.astype(np.float32))
            if not n % 1000:
                print("Processed {}...".format(n))


@hydra.main(config_name="config")
def main(config: DictConfig):
    """Create csv and numpy arrays for a given data directory.

       This will simply create new files if the requested paths don't exist yet,
       and ask if it should overwrite in case they do.

        Parameters:
            config: config object from YAML file.

        """
    want_overwrite = False

    if not os.path.isdir(config.path.root_dir):
        os.mkdir(config.path.root_dir)

    if not os.path.exists(config.path.csv):
        print("The requested corpus csv {} does not seem to exist. "
              "Creating...".format(config.path.csv))
        make_corpus_csv(config.path.raw_dir, config.path.csv)
    else:
        overwrite = input(
            "The requested corpus csv {} exists already. Do you want to "
            "overwrite? This will also overwrite potentially existing "
            "vocabulary files and, most importantly, the mel spectrogram "
            "directory!! Type y/Y to overwrite; anything else will only create "
            "things that don't exist yet.")
        want_overwrite = overwrite.lower() == "y"

    if not os.path.exists(config.path.vocab) or want_overwrite:
        print("Creating vocabulary file {}...".format(config.path.vocab))
        make_vocab(config.path.csv, config.path.vocab)

    if not os.path.isdir(config.path.array_dir) or want_overwrite:
        print("Creating data directory {} (this might take a while)...".format(
            config.path.array_dir))

        resample_rate = config.features.resample_rate
        if resample_rate == 0:
            resample_rate = None
        preprocess_audio(config.path.csv, config.path.raw_dir,
                         config.path.array_dir, resample_rate)


if __name__ == "__main__":
    main()
