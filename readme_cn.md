# CMESS: Towards Controllable Multi-speaker Emotional Speech Synthesis

我们提出了CMESS:一个 Seq2Seq 的情感语音合成模型，能够实现具有高可控性的情感语音合成。相关的音频见[网站](http://isiplab.ahu.edu.cn/etts)

## Pre-requisites

  NVIDIA GPU

## Setup

  1. 下载[ESD数据集](https://hltsingapore.github.io/ESD/) 注意: 你也可以使用其他数据集，但请保持文件格式与现有的一致。(filelists/ESD/esd_train.txt) 文本格式准备部分主要在 “text” 和 “utils” 文件夹中。并不需要对数据集进行预处理。
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
