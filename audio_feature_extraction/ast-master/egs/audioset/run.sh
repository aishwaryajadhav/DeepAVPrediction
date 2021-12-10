#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-5
##SBATCH -p gpu
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast_as"
#SBATCH --output=./log_%j.txt

set -x
# comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
source /ast-master/venvast/bin/activate
export TORCH_HOME=ast-master/pretrained_models

model=ast
dataset=audioset
# full or balanced for audioset
set=full
imagenetpretrain=True
bal=none
lr=5e-5
epoch=25
tr_data=train.json
# else
#   bal=bal
#   lr=1e-5
#   epoch=5
#   tr_data=/data/sls/scratch/yuangong/aed-pc/src/enhance_label/datafiles_local/whole_train_data.json
# fi
te_data=val.json
freqm=48
timem=192
mixup=0.5
# corresponding to overlap of 6 for 16*16 patches
fstride=10
tstride=10
batch_size=32
exp_dir=./exp/test2-${set}-f$fstride-t$tstride-p$imagenetpretrain-b$batch_size-lr${lr}-demo
if [ -d $exp_dir ]; then
  echo 'exp exist'
fi
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ast-master/src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv label_index.csv --n_class 40 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain