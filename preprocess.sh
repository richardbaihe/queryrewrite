#!/usr/bin/env bash

PREFIX_PATH=$(cd `dirname $0`; pwd)
# folder
if [ ! -d temp ];then
          mkdir -p temp
      fi
      if [ ! -d outputs ];then
                mkdir -p outputs
            fi

INPUT=inputs/$1
echo 'concat A and B for $1'
cat ${PREFIX_PATH}/${INPUT}.A ${PREFIX_PATH}/${INPUT}.B > ${PREFIX_PATH}/temp/AB
echo 'lower case; tokenizer; numerify'
python ${PREFIX_PATH}/tools/tokenizer.py < ${PREFIX_PATH}/temp/AB> \
${PREFIX_PATH}/temp/$1.tok &&
echo "get vocab"
python ${PREFIX_PATH}/tools/Vocab/get_vocab.py < ${PREFIX_PATH}/temp/$1.tok \
>${PREFIX_PATH}/temp/$1.vocab.txt

INPUT=inputs/$2
echo 'concat A and B for $2'
cat ${PREFIX_PATH}/${INPUT}.A ${PREFIX_PATH}/${INPUT}.B > ${PREFIX_PATH}/temp/AB
echo 'lower case; tokenizer; numerify'
python ${PREFIX_PATH}/tools/tokenizer.py < ${PREFIX_PATH}/temp/AB> \
${PREFIX_PATH}/temp/$2.tok &&
echo "get vocab"
python ${PREFIX_PATH}/tools/Vocab/get_vocab.py < ${PREFIX_PATH}/temp/$2.tok  \
>${PREFIX_PATH}/temp/$2.vocab.txt

echo "merge vocab"
python ${PREFIX_PATH}/tools/Vocab/merge_vocab.py --vocab_a=${PREFIX_PATH}/temp/$1.vocab.txt \
--vocab_b=${PREFIX_PATH}/temp/$2.vocab.txt --vocab_target=${PREFIX_PATH}/outputs/vocab.txt


echo "get train.unk.en"
python ${PREFIX_PATH}/tools/Vocab/unk.py ${PREFIX_PATH}/temp/$1.tok \
${PREFIX_PATH}/outputs/vocab.txt ${PREFIX_PATH}/temp/$1.unk.txt &&
python ${PREFIX_PATH}/tools/Vocab/unk.py ${PREFIX_PATH}/temp/$2.tok \
${PREFIX_PATH}/outputs/vocab.txt ${PREFIX_PATH}/temp/$2.unk.txt &&
echo "split A and B"
line=$(wc -l < temp/$1.unk.txt)
line=$((${line}/2)) &&
head -n ${line}  ${PREFIX_PATH}/temp/$1.unk.txt > ${PREFIX_PATH}/temp/$1.unk.A &&
tail -n ${line}  ${PREFIX_PATH}/temp/$1.unk.txt > ${PREFIX_PATH}/temp/$1.unk.B &&
line=$(wc -l < temp/$2.unk.txt)
line=$((${line}/2)) &&
head -n ${line}  ${PREFIX_PATH}/temp/$2.unk.txt > ${PREFIX_PATH}/temp/$2.unk.A &&
tail -n ${line}  ${PREFIX_PATH}/temp/$2.unk.txt > ${PREFIX_PATH}/temp/$2.unk.B &&
echo "build train and test set"

cat ${PREFIX_PATH}/temp/$2.unk.A ${PREFIX_PATH}/temp/$2.unk.B ${PREFIX_PATH}/temp/\
$1.unk.A >${PREFIX_PATH}/temp/all.unk.A
cat ${PREFIX_PATH}/temp/$2.unk.B ${PREFIX_PATH}/temp/$2.unk.A ${PREFIX_PATH}/temp/\
$1.unk.B >${PREFIX_PATH}/temp/all.unk.B
line=$(($(wc -l < temp/all.unk.A)-8000)) &&
head -n ${line} ${PREFIX_PATH}/temp/all.unk.A >${PREFIX_PATH}/outputs/train.unk.A
head -n ${line} ${PREFIX_PATH}/temp/all.unk.B >${PREFIX_PATH}/outputs/train.unk.B
tail -n 8000 ${PREFIX_PATH}/temp/all.unk.A >${PREFIX_PATH}/outputs/dev.unk.A
tail -n 8000 ${PREFIX_PATH}/temp/all.unk.B >${PREFIX_PATH}/outputs/dev.unk.B

echo "finished"

