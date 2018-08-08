#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
PREFIX_PATH=$(cd `dirname $0`; pwd)
model=pre_train
STEP=5000
# self-critical?
sc=False
cp ${PREFIX_PATH}/outputs/*vocab ${PREFIX_PATH}/t2t/${model}
cd ${PREFIX_PATH}/t2t/bin &&
#rm ../${model}/model.ckpt* &&

./t2t-datagen \
--data_dir=${PREFIX_PATH}/outputs/tensor2tensor/${model} \
--tmp_dir=${PREFIX_PATH}/outputs/ \
--problem=paraphrase_pretrain \
--num_shards=1

./t2t-trainer \
--self_critical=${sc} \
--worker_gpu=2 \
--hparams='batch_size=1024' \
--data_dir=${PREFIX_PATH}/tensor2tensor/${model} \
--problems=paraphrase_pretrain \
--model=transformer \
--hparams_set=transformer_paraphrase_small \
--train_steps=${STEP} \
--output_dir=${PREFIX_PATH}/tensor2tensor/${model} \
--keep_checkpoint_max=5 \
--save_checkpoints_steps=100
