#!/bin/sh

export CUDA_VISIBLE_DEVICES=""

TEXT="[clears throat] I never really understood why the quick brown fox jumped over a lazy dog? I mean, what was the point? [laughs] Was the fox trying to get eaten?"

tts --model_name "tts_models/multilingual/multi-dataset/bark" \
    --text "$TEXT" \
    --out_path /home/crow/repos/acmp/data/tts-output/hello.wav
    # --use_cuda false
    # --device cuda:0