# CMESS: Towards Controllable Multi-speaker Emotional Speech Synthesis

In this paper, we proposed CMESS: a seq2seq emotional speech synthesis model capable of generating emotional speech synthesis with high controllability. Visit our [website](http://isiplab.ahu.edu.cn/etts) for audio samples.

## Pre-requisites

  NVIDIA GPU

## Setup

  1. Download and extract the [ESD dataset](https://hltsingapore.github.io/ESD/)
     NOTICE: You can also use other data sets, but keep the file format consistent with the existing one. (filelists/ESD/esd_train.txt) The data set preparation part is mainly in the `text` and `utils` folders. There is no need to preprocess the data set.
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

## Other Implementation

  1. git checkout edm / gst
