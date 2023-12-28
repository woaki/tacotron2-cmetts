import sys
import torch
import numpy as np
from scipy.io.wavfile import write

from model import TacotronSTFT
from tacotron2 import Tacotron2
from text import text_to_sequence
from utils import load_wav_to_torch

from hparams import create_hparams
from pypinyin import pinyin, lazy_pinyin, Style


class TTSInfer:
    def __init__(self, hparams, synthesizer_path) -> None:
        self.hparams = hparams
        self.synthesizer = self.get_model(synthesizer_path)
        self.stft = TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            hparams.n_mel_channels,
            hparams.sampling_rate,
            hparams.mel_fmin,
            hparams.mel_fmax,
        )
        pass

    def get_model(self, ckpt_path: str):
        synthesizer = Tacotron2(hparams)
        ckpt = torch.load(ckpt_path)
        synthesizer.load_state_dict(ckpt["state_dict"])
        # vocoder =
        return synthesizer.to("cuda")

    def get_ref_mel(self, ref_audio_path: str):
        audio, sampling_rate = load_wav_to_torch(ref_audio_path)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))
        melspec = self.stft.mel_spectrogram(torch.autograd.Variable(audio.unsqueeze(0), requires_grad=False))
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def get_text(self, text):
        initials = lazy_pinyin(text, neutral_tone_with_five=True, style=Style.INITIALS)
        orig_finals = lazy_pinyin(text, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
        phones = []
        for i, f in zip(initials, orig_finals):
            phones.append(i)
            phones.append(f)

        return " ".join(phones)

    def infer(
        self,
        text: str,
        spk: str,
        ref_audio_path: str = None,
        cg: bool = False,
        std: float = None,
        mean: float = None,
    ):
        phones = self.get_text(text)
        phones = " ".join(phones)
        phones = text_to_sequence(phones, self.hparams.text_cleaners)
        ref_mel = self.get_ref_mel(ref_audio_path)

        phones = torch.LongTensor(phones).unsqueeze(0).to("cuda")
        ref_mel = ref_mel.to("cuda")
        sid = torch.LongTensor([spk]).to("cuda")

        mel = self.synthesizer.inference(phones, sid, ref_mel, cg, std, mean)
        return mel


if __name__ == "__main__":
    hparams = create_hparams()

    tts_infer = TTSInfer(hparams, "Data/v1/ckpt/checkpoint_50000.pt")

    text = "目前的宇宙起源理论认为，宇宙诞生于距今约一百四十亿年前的一次大爆炸"
    spk = 0
    ref_audio = "Data/samples/01-sad.wav"
    tts_infer.infer(text, spk, ref_audio)
