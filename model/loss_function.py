from torch import nn
import torchaudio.functional as F


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    @staticmethod
    def forward(model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_postnet, mel_target)
        # mel_loss = nn.MSELoss()(mel_out, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        loss = mel_loss + gate_loss

        return loss, mel_loss, gate_loss


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    @staticmethod
    def forward(style_out, targets):
        """
        return:
            L_sc -- speaker classification loss
            L_ec -- style classification loss
        """
        _, emotion_cls = style_out
        _, emotion_lab, _ = targets  # [N]

        l_ec = nn.CrossEntropyLoss()(emotion_cls, emotion_lab)

        return l_ec
