#!/bin/bash
python train.py --arch ${1-segnet} --dataset ${2-camvid} --n_epoch ${3-100} --batch_size ${4-8}
