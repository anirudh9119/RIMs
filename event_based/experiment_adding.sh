#!/bin/bash
echo Running on $HOSTNAME
source /home/anirudh/.bashrc
conda activate torch1
run=1
lr=.001
dim1=$1
em=$1
block1=$2
topk1=$3
log=100
train_len=$4
test_len=$5
drop=$6
name="/home/anirudh/RIMs_release/event_based/Blocks_adding/Blocks_"$dim1"_"$em"_"$block1"_"$topk1"_"$drop"_"$lr"_"$log"_"$train_len"_"$test_len
name="${name//./}"
echo Running version $name
python /home/anirudh/RIMs_release/event_based/train_adding.py --cuda --cudnn --algo blocks --name $name --lr $lr --drop $drop --nhid $dim1 --num_blocks $block1 --topk $topk1 --use_inactive --nlayers 1 --emsize $em --log-interval $log --train_len $train_len --test_len $test_len --clip 0.1
