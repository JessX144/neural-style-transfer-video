#!/bin/bash
(set -o igncr) 2>/dev/null && set -o igncr;

python train_transf.py -s ship -w 0.5e1