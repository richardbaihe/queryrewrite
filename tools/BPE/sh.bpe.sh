name=train
#python learn_bpe.py -i ~/corpus/ai_trans/${name}.en -s 80000 -v -o en.80000.codec 
#python apply_bpe.py -i ~/corpus/ai_trans/origin/${name}.en -o ${name}.bpe.en -c en.80.codec &&
#python ~/corpus/BME/unk.py ${name}.bpe.en ~/corpus/ai_trans/en.unk.vocab  ~/corpus/ai_trans/dev.unk.en
python get_vocab.py < ${name}.bpe.en >vocab.en.80
#python apply_bpe.py -i ../quora_data/src_valid.tok -o src_valid.20000 -c pair.train.20000 &&
#python apply_bpe.py -i ../quora_data/src_test.tok -o src_test.20000 -c pair.train.20000 

#python apply_bpe.py -i ../quora_data/tgt_train.tok -o tgt_train.20000 -c pair.train.20000 &&
#python apply_bpe.py -i ../quora_data/tgt_valid.tok -o tgt_valid.20000 -c pair.train.20000 &&
#python apply_bpe.py -i ../quora_data/tgt_test.tok -o tgt_test.20000 -c pair.train.20000 

