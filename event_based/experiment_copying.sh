#!/bin/bash
echo Running on $HOSTNAME
source /home/anirudh/.bashrc
conda activate torch1
lr=.001
dim1=$1
em=$1
block1=$2
topk1=$3
drop=0.5
log=100
train_len=$4
test_len=$5
name="/home/anirudh/RIMs_release/event_based/Blocks_copying/Blocks_"$dim1"_"$em"_"$block1"_"$topk1"_FALSE_"$drop"_"$lr"_"$log"_"$train_len"_"$test_len
name="${name//./}"
echo Running version $name
python /home/anirudh/RIMs_release/event_based/train_copying.py --cuda --cudnn --algo blocks --name $name --lr $lr --drop $drop --nhid $dim1 --num_blocks $block1 --topk $topk1 --nlayers 1 --emsize $em --log-interval $log --train_len $train_len --test_len $test_len
