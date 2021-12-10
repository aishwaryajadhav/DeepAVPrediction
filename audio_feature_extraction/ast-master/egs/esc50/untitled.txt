#!/bin/bash
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ast-esc50"
#SBATCH --output=./log_%j.txt

set -x
# comment this line if not running on sls cluster
% . /data/sls/scratch/share-201907/slstoolchainrc
source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

model=ast
dataset=audioset
imagenetpretrain=True
audiosetpretrain=True
bal=none
if [ $audiosetpretrain == True ]
then
  lr=1e-5
else
  lr=1e-4
fi
freqm=24
timem=192
mixup=0.5
epoch=25
batch_size=64
fstride=10
tstride=10
base_exp_dir=./exp/test-${dataset}-f$fstride-t$tstride-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}

% python3 ./prep_esc50.py

if [ -d $base_exp_dir ]; then
  echo 'exp exist'
else
    mkdir -p $exp_dir
fi

tr_data=train.json
te_data=val.json

CUDA_CACHE_DISABLE=1 python3 -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ./data/esc_class_labels_indices.csv --n_class 50 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain

% python3 ./get_esc_result.py --exp_path ${base_exp_dir}
