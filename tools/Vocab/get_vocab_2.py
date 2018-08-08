#! /usr/bin/env python
import sys
from collections import Counter

c = Counter()

count = 0

for line in sys.stdin:
    gram1 = line.split()
    gram2=[]
    for i in range(len(gram1)):
        if i!=len(gram1)-1:
            gram2.append(gram1[i]+' '+gram1[i+1])
    for word in gram2:
        count += 1
        c[word] += 1

num = 0
words = 0

for key,f in sorted(c.items(), key=lambda x: x[1], reverse=True):
    words += f
    num += 1
    print(key,f)

