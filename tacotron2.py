import torch
import torch.nn.functional as F
from math import sqrt
from torch import nn

from utils import to_gpu, get_mask_from_lengths
from model import Encoder, Decoder, Postnet, StyleDisentangler


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        # Tacotron2 Skeleton
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

        # Speaker ID Embedding
        self.speaker_embedding = nn.Embedding(hparams.num_speakers, hparams.speaker_embedding_dim)

        # Emotion Encoder
        self.disentangler = StyleDisentangler(hparams, hparams.ref_enc_gru_size, hparams.num_emotions)

    @staticmethod
    def parse_batch(batch, hparams):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, speaker_id, emotion_id = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        speaker_id = to_gpu(speaker_id).long()
        emotion_lab = to_gpu(F.one_hot(emotion_id, hparams.num_emotions)).float()
        # speaker_lab = to_gpu(F.one_hot(speaker_id, hparams.num_speakers)).float()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths, speaker_id),
            (mel_padded, gate_padded),
            (emotion_id, emotion_lab, speaker_id),
        )

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        """
        inputs:
            text_inputs : [N, T_l]
            text_length : [N]
            mels : [N, M_c, M_l]
            max_len : max_text_length (int)
            outputs_lengths : [N] --> mel_lengths
        outputs:
            mel_outputs -- [N, M_c, M_l]
            gate_outputs -- [N, M_l]
            alignments -- [N, M_l, T_l]
            mel_outputs_postnet -- [N, M_c, M_l]
        """
        text_inputs, text_lengths, mels, max_len, output_lengths, speaker_id = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        # embedded
        embedded_text = self.embedding(text_inputs).transpose(1, 2)  # [N, 512, t_l]
        embedded_speaker = self.speaker_embedding(speaker_id)  # [N, 32]
        emotion_embedding, emo_cls = self.disentangler(mels, output_lengths)

        # get decoder input ready
        encoder_outputs = self.encoder(embedded_text, text_lengths)  # [N, T_l, 512]
        encoder_outputs = encoder_outputs + embedded_speaker.unsqueeze(1)
        emotion_input = emotion_embedding.unsqueeze(1).repeat(1, embedded_text.size(2), 1)  # [N, T_l, 128]
        decoder_inputs = torch.cat((encoder_outputs, emotion_input), dim=2)  # [n, T_l, 512 + 128 + 32]

        # Tacotron2 Decoder
        mel_outputs, gate_outputs, alignments = self.decoder(
            decoder_inputs, mels, memory_lengths=text_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        parsed_out = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments], output_lengths
        )
        style_out = (emotion_embedding, emo_cls)

        return parsed_out, style_out, alignments

    def inference(self, text, speaker_id, mels, cg=False, std=None, mean=None):
        # Reference audio mel
        mels = mels.unsqueeze(0)  # [1, m_c, m_l]
        # embedded text
        embedded_text = self.embedding(text).transpose(1, 2)
        embedded_speaker = self.speaker_embedding(speaker_id)
        emotion_embedding = self.disentangler.inference(mels)
        if cg:
            emotion_embedding = self.change_style(emotion_embedding, std, mean)

        # get Decoder inputs already
        encoder_outputs = self.encoder.inference(embedded_text)
        emotion_input = emotion_embedding.unsqueeze(1).repeat(1, embedded_text.size(2), 1)
        speaker_input = embedded_speaker.unsqueeze(1).repeat(1, embedded_text.size(2), 1)
        decoder_inputs = torch.cat((encoder_outputs, emotion_input, speaker_input), dim=2)
        # Decoder
        mel_outputs, gate_outputs, alignments = self.decoder.inference(decoder_inputs)

        # Postnet
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs

    @staticmethod
    def change_style(feature, std, mean):
        feature_mean = feature.mean()
        feature_std = feature.std()
        new_feature = std * (feature - feature_mean) / feature_std + mean
        return new_feature
