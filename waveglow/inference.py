import os
from scipy.io.wavfile import write
import torch
from mel2samp import files_to_list, MAX_WAV_VALUE
from .denoiser import Denoiser


def main(mel_files, waveglow_path, sigma, output_dir, sampling_rate, is_fp16,
         denoiser_strength):
    mel_files = files_to_list(mel_files)
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()

    for i, file_path in enumerate(mel_files):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        mel = torch.load(file_path)
        mel = torch.autograd.Variable(mel.cuda())
        mel = torch.unsqueeze(mel, 0)
        mel = mel.half() if is_fp16 else mel
        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=sigma)
            if denoiser_strength > 0:
                audio = denoiser(audio, denoiser_strength)
            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        audio_path = os.path.join(
            output_dir, "{}_synthesis.wav".format(file_name))
        write(audio_path, sampling_rate, audio)
        print(audio_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    main(args.filelist_path, args.waveglow_path, args.sigma, args.output_dir,
         args.sampling_rate, args.is_fp16, args.denoiser_strength)
