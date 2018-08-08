import numpy
import pickle as pkl

import sys
import fileinput

from collections import OrderedDict

def main():
    for filename in sys.argv[1:]:
        print ('Processing', filename)
        word_freqs = OrderedDict()
        count = 0
        with open(filename, 'r',encoding='utf-8') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    count += 1
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
        words = list(word_freqs.keys())
        freqs = list(word_freqs.values())
        #freqs = [i if int(i) > 3 for i in freqs]

        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]
        sorted_freqs = [freqs[ii] for ii in sorted_idx[::-1]]  ##

        worddict = OrderedDict()
        worddict['eos'] = 0
        worddict['UNK'] = 1
        num = 0
        for ii, ww in enumerate(sorted_words):
            if ii <= 30000:
            #if sorted_freqs[ii] > 9:
                #worddict[ww] = ii+2
                worddict[ww] = ii + 2
                num += sorted_freqs[ii]
                #num += 1
        
        #with open('%s.freq10.pkl'%filename, 'wb') as f:
        #    pkl.dump(worddict, f)
       
        print( 'lowest freqs: ', sorted_freqs[30000])

        print ('count: ', count)
        print ('num: ', num)
        print ('num/count: ', (num*1.0)/count)

        print ('Done')

if __name__ == '__main__':
    main()
