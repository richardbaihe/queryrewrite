#! /usr/bin/env python
import argparse,sys

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description="learn BPE-based word segmentation")

parser.add_argument('--input','-i',type=argparse.FileType('r',encoding='utf-8'),default=sys.stdin,metavar='PATH',help="Input text (default: standard input).")
parser.add_argument('--output', '-o', type=argparse.FileType('w',encoding='utf-8'), default=sys.stdout, metavar='PATH',help="Output file for BPE codes (default: standard output)")
parser.add_argument('--symbols', '-s', type=int, default=10000,help="Create this many new symbols (each representing acharacter n-gram) (default: %(default)s))")

args = parser.parse_args()
num = 0
all_lines = args.input.readlines()
if args.symbols==0:
    args.symbols=len(all_lines)
for line in all_lines:
    if num == args.symbols:
        break
    args.output.write(' '.join(line.split()[:-1])+'\n')
    num += 1
args.output.write('UNK\n')
