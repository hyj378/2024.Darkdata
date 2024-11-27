#!/usr/bin/env bash
SAVEDIR=$1

python train.py configs/vfa/voc/vfa_split1/vfa_r101_c4_8xb4_voc-split1_10shot-fine-tuning_JB.py --work-dir $SAVEDIR