import os
import time
import argparse
from numpy import finfo

import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader

from tacotron2 import Tacotron2
from utils import TextMelLoader, TextMelCollate
from model import Tacotron2Loss, StyleLoss
from logger import Tacotron2Logger
from hparams import create_hparams


def prepare_dataloaders(_hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(_hparams.training_files, _hparams)
    valset = TextMelLoader(_hparams.validation_files, _hparams)
    collate_fn = TextMelCollate(_hparams)

    train_sampler = None
    shuffle = True

    # text_padded, input_lengths, mel_padded, gate_padded, output_lengths
    train_loader = DataLoader(
        trainset,
        num_workers=64,
        shuffle=shuffle,
        sampler=train_sampler,
        batch_size=_hparams.batch_size,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    logger = Tacotron2Logger(log_directory)
    return logger


def load_model(_hparams):
    model = Tacotron2(_hparams).cuda()
    if _hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo("float16").min

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = checkpoint_dict["state_dict"]
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    learning_rate = checkpoint_dict["learning_rate"]
    iteration = checkpoint_dict["iteration"]
    print("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(iteration, filepath))
    torch.save(
        {
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        filepath,
    )


def validate(hparams, model, criterion, style_criterion, valset, iteration, collate_fn, logger):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(
            valset,
            sampler=None,
            num_workers=1,
            shuffle=False,
            batch_size=hparams.batch_size,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y, style_targets = model.parse_batch(batch, hparams)
            y_pred, style_out, alignments = model(x)
            tacotron_loss, _, _ = criterion(y_pred, y)
            style_loss = style_criterion(style_out, style_targets)
            loss = tacotron_loss + style_loss
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

    model.train()

    print("Validation loss {}: {:9f}".format(iteration, val_loss))
    logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(model_directory, log_directory, checkpoint_path, warm_start, hparams, pretrain=False):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    model_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.

    """
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    tacotron2 = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(tacotron2.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay)

    # criterion
    criterion = Tacotron2Loss()
    style_criterion = StyleLoss()

    logger = prepare_directories_and_logger(model_directory, log_directory)

    train_loader, val_set, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            tacotron2 = warm_start_model(checkpoint_path, tacotron2, hparams.ignore_layers)
        else:
            tacotron2, optimizer, learning_rate, iteration = load_checkpoint(
                checkpoint_path, tacotron2, optimizer
            )
            if hparams.use_saved_learning_rate:
                learning_rate = learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    tacotron2.train()
    is_overflow = False
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            tacotron2.zero_grad()

            # get training data ready
            x, y, style_targets = tacotron2.parse_batch(batch, hparams)

            # model outputs
            y_pred, style_out, alignments = tacotron2(x)

            # figure up loss
            tacotron_loss, mel_loss, gate_loss = criterion(y_pred, y)
            if not pretrain:
                style_loss = style_criterion(style_out, style_targets)
                loss = tacotron_loss + style_loss
            else:
                loss = tacotron_loss

            # backward & update
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(tacotron2.parameters(), hparams.grad_clip_thresh)
            optimizer.step()

            # logging
            if not is_overflow:
                duration = time.perf_counter() - start
                print(
                    "Train loss {} {:.6f} Style loss {:.6f} {:.2f}s/it".format(
                        iteration, loss, style_loss, duration
                    )
                )
                logger.log_training(
                    loss.item(),
                    mel_loss.item(),
                    gate_loss.item(),
                    style_loss,
                    grad_norm,
                    learning_rate,
                    duration,
                    iteration,
                    alignments,
                )

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))  # recording time
                validate(tacotron2, criterion, style_criterion, val_set, iteration, collate_fn, logger)
                # saving checkpoint
                checkpoint_path = os.path.join(model_directory, "checkpoint_{}.pt".format(iteration))
                save_checkpoint(tacotron2, optimizer, learning_rate, iteration, checkpoint_path)

            iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        help="directory to save checkpoints",
        default="./Data/your_data/ckpt",
    )
    parser.add_argument(
        "-l",
        "--log_directory",
        type=str,
        help="directory to save tensorboard logs",
        default="./Data/your_data/logs",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="checkpoint path",
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="load model weights only, ignore specified layers",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="pretrain or not",
    )
    args = parser.parse_args()
    hparams = create_hparams()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(
        args.output_directory,
        args.log_directory,
        args.checkpoint_path,
        args.warm_start,
        hparams,
        args.pretrain,
    )
