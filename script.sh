#!/bin/bash
(set -o igncr) 2>/dev/null && set -o igncr;

python train_transf.py -s splotch -w 0.3e1
python train_transf.py -s splotch -w 0.5e1