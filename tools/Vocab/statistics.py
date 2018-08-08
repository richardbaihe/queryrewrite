import sys
total = []
for line in sys.stdin:
    total.append(int(line.split()[1]))
i=5
while i < 40 :
    temp = [t for t in total if t<=i]
    print('freq less than {0} : {1}%, {2}'.format(i,len(temp)/len(total),len(temp)))
    i+=5
