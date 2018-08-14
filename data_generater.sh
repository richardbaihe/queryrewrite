#!/usr/bin/env bash
PREFIX_PATH=$(cd `dirname $0`; pwd)
model=adapt

cd ${PREFIX_PATH}/tensor2tensor/bin &&
./t2t-datagen \
--data_dir=${PREFIX_PATH}/${model} \
--tmp_dir=${PREFIX_PATH}/${model} \
--problem=paraphrase_pretrain \
--num_shards=1
