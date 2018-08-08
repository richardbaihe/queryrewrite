import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,\
                                 description='return new words in vocab a compared to vocab b')
parser.add_argument('--vocab_a','-a',type=argparse.FileType('r',encoding='utf-8'),metavar='PATH')
parser.add_argument('--vocab_b','-b',type=argparse.FileType('r',encoding='utf-8'),metavar='PATH')
parser.add_argument('--vocab_diff','-d',type=argparse.FileType('w',encoding='utf-8'),metavar='PATH')
args = parser.parse_args()

dic_a = {}
for line in args.vocab_a:
    word, freq = line.split()
    dic_a[word]=freq
dic_b = []
for word in args.vocab_b:
    dic_b.append(word.split()[0])

oov = {}
for word,freq in dic_a.items():
    if word in dic_b:
        continue
    oov[word]=int(freq)
    args.vocab_diff.write(word+'\t'+str(freq)+'\n')
    
