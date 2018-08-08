import argparse,sys

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,\
                                 description='extract oov lines from files')
parser.add_argument('--oov_a','-a',type=argparse.FileType('r',encoding='utf-8'),metavar='PATH')
parser.add_argument('--oov_b','-b',type=argparse.FileType('r',encoding='utf-8'),metavar='PATH')
parser.add_argument('--oov_c','-c',type=argparse.FileType('w',encoding='utf-8'),metavar='PATH')
parser.add_argument('--input','-i',type=argparse.FileType('r',encoding='utf-8'),metavar='PATH')
parser.add_argument('--output','-o',type=argparse.FileType('w',encoding='utf-8'),metavar='PATH')
args = parser.parse_args()

oov_a = []
for line in args.oov_a:
    word, freq = line.split()
    oov_a.append(word)
oov_b = []
for line in args.oov_b:
    word, freq = line.split()
    oov_b.append(word)

oov = []
for word in oov_a:
    if word in oov_b:
        oov.append(word)
    args.oov_c.write(word+'\n')
for line_a,line_b in zip(args.input,sys.stdin):
    for word in line_a.split():
        if word in oov:
            args.output.write(line_a)
            print(line_b[:-1])
            break
