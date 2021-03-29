#!/bin/bash
(set -o igncr) 2>/dev/null && set -o igncr;

python train.py -s face 
python train_noise.py -s bw 
python train_flow.py -s flower 
python stylise.py -s picasso -c bo
python stylise.py -s picasso -n VIDEO_NAME -u YOUTUBE_URL