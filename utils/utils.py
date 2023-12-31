import numpy as np

# from scipy.io.wavfile import read
import torch
import librosa


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    # ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).cuda())
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    # sampling_rate, data = read(full_path)
    data, sampling_rate = librosa.load(full_path, sr=16000)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, "r", encoding="utf-8") as f:
        # filepaths_and_text = [[filepath, text], ....]
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
