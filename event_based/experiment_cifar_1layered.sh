#!/bin/bash
echo Running on $HOSTNAME
lr=.0007
dim1=$1
em=$1
block1=$2
topk1=$3
drop=0.2
log=100
name="Blocks_Cifar/Blocks_"$dim1"_"$em"_"$block1"_"$topk1"_FALSE_"$drop"_"$lr"_"$log
name="${name//./}"
echo Running version $name
python3 train_cifar.py --cuda --cudnn --algo blocks --name $name --lr $lr --drop $drop --nhid $dim1 --num_blocks $block1 --topk $topk1 --nlayers 1 --emsize $em --log-interval $log
