import sys, re


def tokenizer_char(txt):
    def match_num(matched):
        begin, end = matched.regs[0]
        length = str(end-begin)
        return ' #num'+length+' '

    def match_en(matched):
        begin, end = matched.regs[0]
        word = matched.string[begin:end]
        if len(word) > 1:
            return ' ' + word + ' '
        else:
            return ''
    txt = txt.lower()
    txt = re.sub(u'[!“\"#$%&\'()+,-./:;<=>?@[\]^_`{|}~，。！？、【】「」～]+', '', txt)
    txt = re.sub(u'[a-zA-z]+', match_en, txt)
    txt = re.sub(u'[0-9]+\*+[0-9]+|[0-9]+|\*\*\*', match_num, txt)
    txt = re.sub(u'[\u4e00-\u9fa5]+', ' #CH ', txt)
    txt = re.sub('\s+', ' ', txt)
    return txt

for line in sys.stdin:
    result = tokenizer_char(line)
    print(result)

