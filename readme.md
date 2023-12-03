# CMESS: Towards Controllable Multi-speaker Emotional Speech Synthesis

In this paper, we proposed CMESS: a seq2seq emotional speech synthesis model capable of generating emotional speech synthesis with high controllability. Visit our [website](https://isiplabahu.github.io/cmetts) for audio samples.

## Pre-requisites

  NVIDIA GPU

## Setup

  1. Download and extract the [ESD dataset](https://hltsingapore.github.io/ESD/)
     ```
     wav_path|phones|speaker_id|emotion_category
     ```
  2. Install pytorch and python requirements
  3. Get train-sets ready
  4. run train.py

## Inference

  1. Get CMESS model ready
  2. Get [WaveGlow](https://github.com/NVIDIA/waveglow) model ready
  3. load inference.ipynb
