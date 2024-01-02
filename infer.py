import torch
import json
from scipy.io.wavfile import write

from pypinyin import pinyin, lazy_pinyin, Style

from text import text_to_sequence
from utils import load_wav_to_torch
from model import TacotronSTFT
from tacotron2 import Tacotron2
from hifigan import Generator

from hparams import create_hparams

device = "cuda" if torch.cuda.is_available() else "cpu"


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class TTSInfer:
    def __init__(self, hparams, ckpt_paths: list, config_path: str) -> None:
        self.hparams = hparams
        self.synthesizer, self.vocoder = self.get_model(ckpt_paths, self.get_json(config_path))
        with open(config_path) as f:
            data = f.read()

        json_config = json.loads(data)
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

    def get_json(self, config_path: str):
        with open(config_path) as f:
            data = f.read()

        json_config = json.loads(data)
        return AttrDict(json_config)

    def get_model(self, _ckpt_path: str, v_config):
        synthesizer = Tacotron2(hparams)
        vocoder = Generator(v_config)

        syn_ckpt = torch.load(_ckpt_path[0], map_location=torch.device(device))
        vocoder_ckpt = torch.load(_ckpt_path[1], map_location=torch.device(device))
        synthesizer.load_state_dict(syn_ckpt["state_dict"])
        vocoder.load_state_dict(vocoder_ckpt["generator"])

        # vocoder =
        return synthesizer.to(device), vocoder.to(device)

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
        phones = text_to_sequence(" ".join(phones), self.hparams.text_cleaners)

        return phones

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
        ref_mel = self.get_ref_mel(ref_audio_path)

        phones = torch.LongTensor(phones).unsqueeze(0).to(device)
        ref_mel = ref_mel.to(device)
        sid = torch.LongTensor([spk]).to(device)

        with torch.no_grad():
            _, mel = self.synthesizer.inference(phones, sid, ref_mel, cg, std, mean)
            y_g_hat = self.vocoder(mel)
            audio = y_g_hat.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype("int16")

        return audio


if __name__ == "__main__":
    hparams = create_hparams()

    ckpt_paths = ["Data/cakpt/ta.pt", "hifigan/ckpt/g.pt"]
    tts_infer = TTSInfer(hparams, ckpt_paths, "hifigan.json")

    text = "目前的宇宙起源理论认为，宇宙诞生于距今约一百四十亿年前的一次大爆炸"
    spk = 0
    ref_audio = "samples/01-sad.wav"
    audio = tts_infer.infer(text, spk, ref_audio)
    write("demo.wav", 16000, audio)
