#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
PREFIX_PATH=$(cd `dirname $0`; pwd)
model=pretrain
STEP=10000
# self-critical?
sc=False

cd ${PREFIX_PATH}/tensor2tensor/bin &&
#rm -rf ${PREFIX_PATH}/tensor2tensor/${model} &&
mkdir ${PREFIX_PATH}/tensor2tensor/${model}

./t2t-trainer \
--self_critical=${sc} \
--worker_gpu=1 \
--hparams='batch_size=20480' \
--data_dir=${PREFIX_PATH}/${model}/ \
--problems=paraphrase_pretrain \
--model=transformer \
--hparams_set=transformer_paraphrase_small \
--train_steps=${STEP} \
--output_dir=${PREFIX_PATH}/tensor2tensor/${model} \
--keep_checkpoint_max=5 \
--save_checkpoints_steps=300 \
--local_eval_frequency=500 \
--decode_beam_size=1 \
--decode_batch_size=512 \
--decode_alpha=1.0 \
--decode_return_beams=False \
--decode_from_file=${PREFIX_PATH}/${model}/dev.unk.A \
--decode_to_file=${PREFIX_PATH}/${model}/result.txt


#./t2t-trainer \
#--self_critical=${sc} \
#--worker_gpu=1 \
#--data_dir=${PREFIX_PATH}/tensor2tensor/${model} \
#--problems=paraphrase_pretrain \
#--model=transformer \
#--hparams_set=transformer_paraphrase_small \
#--output_dir=${PREFIX_PATH}/tensor2tensor/${model} \
#--train_steps=0 \
#--eval_steps=0 \
#---decode_alpha=1.0 \
#--decode_return_beams=False \
#--decode_from_file=${PREFIX_PATH}/outputs/dev.unk.A \
#--decode_to_file=${PREFIX_PATH}/outputs/result.txt
