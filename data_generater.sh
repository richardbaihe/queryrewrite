#!/usr/bin/env bash
PREFIX_PATH=$(cd `dirname $0`; pwd)
model=pre_train
mkdir ${PREFIX_PATH}/tensor2tensor/${model}
cp ${PREFIX_PATH}/outputs/vocab.txt ${PREFIX_PATH}/tensor2tensor/${model}/

cd ${PREFIX_PATH}/tensor2tensor/bin &&
./t2t-datagen \
--data_dir=${PREFIX_PATH}/tensor2tensor/${model} \
--tmp_dir=${PREFIX_PATH}/outputs/ \
--problem=paraphrase_pretrain \
--num_shards=1
