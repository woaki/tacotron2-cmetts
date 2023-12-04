from waveglow.denoiser import Denoiser
import sys
import torch
import numpy as np
from scipy.io.wavfile import write

from model import TacotronSTFT
from tacotron2 import Tacotron2
from text import text_to_sequence
from utils import load_wav_to_torch

from hparams import create_hparams

hparams = create_hparams()

sys.path.append("waveglow")

from hparams import create_hparams
from g2pM import G2pM


def get_model():
    return None


def get_vocoder():
    return None


if __name__ == "__main__":
    model = get_model()
    vocoder = get_vocoder()
