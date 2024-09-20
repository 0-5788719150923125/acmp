#!/bin/sh

export CUDA_VISIBLE_DEVICES=""

tts --model_name "tts_models/multilingual/multi-dataset/bark" \
    --text "Hello." \
    --out_path /home/crow/repos/acmp/data/tts-output/hello.wav
    # --use_cuda false
    # --device cuda:0