import os
import random
import argparse
import json
import librosa
import torch
import torch.utils.data
import numpy as np
import torch.nn.functional

# We're using the audio processing from TacoTron2 to make sure it matches
from .utils import TacotronSTFT

MAX_WAV_VALUE = 32768.0


def files_to_list(_filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(_filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def load_wav_to_torch(full_path):
    """
    Loads wav_data into torch array
    """
    # sampling_rate, _data = load(full_path)
    # _data = torch.FloatTensor(_data.astype(np.float32))
    _data, sampling_rate = librosa.load(full_path, sr=16000)
    return torch.FloatTensor(_data.astype(np.float32)), sampling_rate
    # return _data, sampling_rate


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        self.audio_files = files_to_list(training_files)
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_mel(self, _audio):
        # audio_norm = _audio / MAX_WAV_VALUE
        # audio_norm = audio_norm.unsqueeze(0)
        # audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        # _audio = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(
            torch.autograd.Variable(_audio.unsqueeze(0), requires_grad=False)
        )
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        _filename = self.audio_files[index]
        _audio, sampling_rate = load_wav_to_torch(_filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        # Take segment
        if _audio.size(0) >= self.segment_length:
            max_audio_start = _audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            _audio = _audio[audio_start:audio_start + self.segment_length]
        else:
            _audio = torch.nn.functional.pad(_audio, (0, self.segment_length - _audio.size(0)), 'constant').data

        mel = self.get_mel(_audio)
        # _audio = _audio / MAX_WAV_VALUE

        return mel, _audio

    def __len__(self):
        return len(self.audio_files)


# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    mel2samp = Mel2Samp(**data_config)

    filepaths = files_to_list(args.filelist_path)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    for filepath in filepaths:
        audio, sr = load_wav_to_torch(filepath)
        mel_spectrogram = mel2samp.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(mel_spectrogram, new_filepath)
