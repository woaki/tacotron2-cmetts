import random
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

from model import TacotronSTFT
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
    1) loads audio,text pairs
    2) normalizes text and converts them to sequences of one-hot vectors
    3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)  # [[filepath, text], ....]
        self.text_cleaners = hparams.text_cleaners
        # self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            hparams.n_mel_channels,
            hparams.sampling_rate,
            hparams.mel_fmin,
            hparams.mel_fmax,
        )
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_data(self, audiopath_and_text):
        audiopath, text, speaker, emotion = audiopath_and_text
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        speaker_id = self.get_speaker_id(speaker)
        emotion_id = self.get_emotion_id(emotion)
        return text, mel, speaker_id, emotion_id

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return text, mel

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError(
                    "{} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate)
                )
            # audio_norm = audio / self.max_wav_value
            # audio_norm = audio_norm.unsqueeze(0)
            # audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            # melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = self.stft.mel_spectrogram(
                torch.autograd.Variable(audio.unsqueeze(0), requires_grad=False)
            )
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert (
                melspec.size(0) == self.stft.n_mel_channels
            ), "Mel dimension mismatch: given {}, expected {}".format(
                melspec.size(0), self.stft.n_mel_channels
            )

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    @staticmethod
    def get_speaker_id(speaker):
        """
        Args:
            speaker: str -- speaker
        Returns:
            speaker_id: Tensor
        """
        if speaker == "bznsy":
            speaker = "0011"
        speaker_id = torch.IntTensor([int(speaker.strip()) - 1])
        return speaker_id

    @staticmethod
    def get_emotion_id(emotion):
        """
        Args:
            emotion: str -- style
        Returns:
            emotion_id: Tensor
        """
        # emotions = ["angry", "happy", "surprise", "neutral", "sad"]
        emotions = ["生气", "快乐", "惊喜", "中立", "伤心"]
        emotion_id = torch.IntTensor([emotions.index(emotion)])
        return emotion_id

    def __getitem__(self, index):
        return self.get_data(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate:
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, hparams):
        self.n_frames_per_step = hparams.n_frames_per_step
        self.num_emotions = hparams.num_emotions
        self.num_speakers = hparams.num_speakers

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len).zero_()  # [N, max_text_len]
        # text_padded.zero_()
        speaker_id = torch.LongTensor(len(batch))
        emotion_id = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            # get the corresponding speaker_id and emotion_id
            speaker_id[i] = batch[ids_sorted_decreasing[i]][2]
            emotion_id[i] = batch[ids_sorted_decreasing[i]][3]
            # get text_padded
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)

        return (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            speaker_id,
            emotion_id,
        )
