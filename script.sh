#!/bin/bash
(set -o igncr) 2>/dev/null && set -o igncr;

python train.py -s picasso 
python train_noise.py -s picasso 
python train_flow.py -s picasso 
python stylise.py -s picasso -c bo