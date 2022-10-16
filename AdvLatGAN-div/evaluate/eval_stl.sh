#!/bin/bash
set -x
for item in airplane car bird cat deer dog monkey horse ship truck all; do
  python ./eval.py --metric fid --pred_list $1/$item.txt --gt_list $2/$item.txt --gpu_id $3 --resize 299
done