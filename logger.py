import random
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, loss, mel_loss, gate_loss, style_loss,
                     grad_norm, learning_rate, duration, iteration, alignments):
        self.add_scalar("training.loss", loss, iteration)

        self.add_scalar("mel.loss", mel_loss, iteration)
        self.add_scalar("gate.loss", gate_loss, iteration)
        # self.add_scalar("delta.loss", delta_loss, iteration)
        self.add_scalar("style.loss", style_loss, iteration)

        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)
        if iteration % 2000 == 0:
            self.add_image(
                "training_alignment",
                plot_alignment_to_numpy(alignments[0].data.cpu().numpy().T),
                iteration, dataformats='HWC'
            )

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plotting_function distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plotting_function alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "validation_alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
