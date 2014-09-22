import numpy as np
from Doc import Doc

class Corpus(object):
    def __init__(self,filename):
        self.docs = []
        self.vocab_size = 0
        self.n_docs = 0
        print "reading data from : "+filename
        f=open(filename)
        for line in f.readlines():
            parts=line.split()
            doc=Doc()
            for i,part in enumerate(parts):
                if i==0:
                    doc.length=int(part)
                else:
                    word,count=map(int,part.split(":"))
                    doc.words.append(word)
                    doc.word_counts.append(count)
                    doc.total+=count
                    if(word>=self.vocab_size):
                        self.vocab_size=word+1
        f.close()
        self.n_docs=len(self.docs)




