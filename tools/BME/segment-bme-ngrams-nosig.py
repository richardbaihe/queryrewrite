#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

from __future__ import unicode_literals, division

import sys
#import pdb
import codecs
import argparse

# hack for python2/3 compatibility
from io import open
argparse.open = open

# python 2/3 compatibility
if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="segment rare words into character n-grams")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--vocab', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="Vocabulary file.")
    parser.add_argument(
        '--shortlist', type=int, metavar='INT', default=0,
        help="do not segment INT most frequent words in vocabulary (default: '%(default)s')).")
    parser.add_argument(
        '-n', type=int, metavar='INT', default=2,
        help="segment rare words into character n-grams of size INT (default: '%(default)s')).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")

    return parser


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

    vocab = [line.split()[0] for line in args.vocab if len(line.split()) == 2]
    #vocab = [line.split()[0] for line in args.vocab]
    
    '''
    vocab = []
    for line in args.vocab:
        if len(line.split()) == 2:
            vocab.append(line.split()[0])
        else:
            print 'error: ', line.strip()
    '''

    #print vocab[53110] ##cn
    ##print vocab[52173] ## jp
    ##for i in range(10):
    ##    print i,': ', vocab[i]
    #print vocab[50000]
    #vocab = dict((y,x) for (x,y) in enumerate(vocab) if x <= args.shortlist)
    
    Vocab = dict()
    for x, y in enumerate(vocab):
        if x <= args.shortlist:
            Vocab[y] = x 
    
    ##for k,v in Vocab.items():
    ##    if int(v) < 10:
    ##        print k, ' ', v
    #print vocab['印存储']
    #print vocab['中 ']
    #print Vocab['中']
    #print Vocab['例']
    
    #pdb.set_trace()
    for line in args.input:
      for word in line.split():
        #if word not in vocab or vocab[word] > args.shortlist:
        if word not in Vocab:
          #args.output.write('UNK ')
          for i in range(len(word)):
              if i == 0:
                  args.output.write(word[i] + ' ')
              elif i==len(word)-1:
                  args.output.write(word[i] + ' ')
              else:
                  args.output.write(word[i] + ' ')

          '''
          i = 0
          while i*args.n < len(word):
            args.output.write(word[i*args.n:i*args.n+args.n])
            i += 1
            if i*args.n < len(word):
              args.output.write(args.separator)
            args.output.write(' ')
          '''
        else:
          args.output.write(word + ' ')
      args.output.write('\n')
