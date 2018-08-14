#! /usr/bin/env python
import argparse,sys

parser = argparse.ArgumentParser()

parser.add_argument('--vocab_a',type=str,required=True)
parser.add_argument('--vocab_b', type=str,required=True)
parser.add_argument('--vocab_target', type=str,required=True)

args = parser.parse_args()

# vocab_a = args.vocab_a.readlines()
vocab_target = open(args.vocab_target,'w',encoding='utf-8')
vocab_a = open(args.vocab_a,'r',encoding='utf-8').readlines()
vocab = {}
for line in vocab_a:
    word,num = line.strip().split()
    if int(num) < 5:
        break
    vocab[word] = num
    #vocab_target.write(line)

vocab_b = open(args.vocab_b,'r',encoding='utf-8').readlines()
for line in vocab_b:
    word,num = line.strip().split()
    if len(vocab) > 30000 or int(num) < 10:
        break
    vocab[word] = num

for word,freq in vocab.items():
    vocab_target.write(word+'\n')
vocab_target.write('UNK')
vocab_target.close()
