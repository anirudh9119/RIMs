#!/bin/bash
echo Running on $HOSTNAME
run=1
lr=.0007
dim1=$1
em=$1
block1=$2
topk1=$3
drop=0.5
log=100
name="Blocks_MNIST/Blocks_"$dim1"_"$em"_"$block1"_"$topk1"_"$drop"_"$lr"_"$log
name="${name//./}"
echo Running version $name
python3 train_mnist.py --cuda --cudnn --algo blocks --name $name --lr $lr --drop $drop --nhid $dim1 --num_blocks $block1 --topk $topk1 --nlayers 1 --emsize $em --log-interval $log
