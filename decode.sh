#export CUDA_VISIBLE_DEVICES=3
PREFIX_PATH=$(cd `dirname $0`; pwd)
model=adapt
result=result_sl_200.txt
cd ${PREFIX_PATH}/t2t/bin &&
./t2t-trainer --locally_shard_to_cpu=True --worker_gpu=0 --self_critical=False --data_dir=${PREFIX_PATH}/t2t/${model} --problems=wmt_ende_bpe32k --model=transformer --hparams_set=transformer_big_single_gpu --output_dir=${PREFIX_PATH}/t2t/${model} --train_steps=0 --eval_steps=0 --decode_beam_size=4 --decode_batch_size=30 --decode_alpha=1.0 --decode_return_beams=False --decode_from_file=${PREFIX_PATH}/outputs/sc.unk.sc --decode_to_file=${PREFIX_PATH}/temp/${result}
sed -i "s/@@ //g" ${PREFIX_PATH}/temp/${result}  &&
sed -i "s/&apos;/'/g" ${PREFIX_PATH}/temp/${result}

cp ${PREFIX_PATH}/temp/${result} ${PREFIX_PATH}/temp/result.txt &&
cd ${PREFIX_PATH}/tools/ &&
./slot_filled.py &&
cd ${PREFIX_PATH}/temp/ &&
paste -d '\t' domain.tsv intent.tsv replaced_result.txt > ../outputs/result.tsv
