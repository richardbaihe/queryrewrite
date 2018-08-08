#!/usr/bin/env bash
python tensor2tensor/bin/t2t-datagen --data_dir=./tensor2tensor/pretrain/ --tmp_dir=./outputs/ --problem=paraphrase_pretrain --num_shards=1