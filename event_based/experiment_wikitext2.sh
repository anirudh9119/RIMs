#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
source activate torch1
lr=$1
dim1=$2
em=$2
block1=$3
topk1=$4
drop=$5
log=100
name="/home/anirudh/icml_blocks/sparse_relational/Blocks_wiki2/experimental_Blocks_wiki2_"$dim1"_"$em"_"$block1"_"$topk1"_FALSE_"$drop"_"$lr"_"$log
name="${name//./}"
echo Running version $name


#./experiment_wikitext2.sh 0.0007 510 510 5 5 0.5. It should get a test ~ 102.

#To run, 1 layered RIM baseline
python /home/anirudh/icml_blocks/sparse_relational/train_wiki.py --tied --cuda --cudnn --algo blocks --data /home/anirudh/icml_blocks/sparse_relational/data/wikitext-2 --name $name --lr $lr --drop $drop --nhid $dim1  --num_blocks $block1  --topk $topk1 --use_inactive --nlayers 1 --emsize $em --log-interval $log



#To run 2 layered RIM baseline.
#python /home/anirudh/icml_blocks/sparse_relational/train_wiki.py --tied --cuda --cudnn --algo blocks --data /home/anirudh/icml_blocks/sparse_relational/data/wikitext-2 --name $name --lr $lr --drop $drop --nhid $dim1 $dim1 --num_blocks $block1 $block1 --topk $topk1 $topk1 --use_inactive --nlayers 2 --emsize $em --log-interval $log
