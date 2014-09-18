import numpy as np
from scipy import log
import numpy.random as rnd


class LdaModel(object):
    def __init__(self, vocab_size, n_topics,initType=None,corpus=None):
        self.NUM_INIT = 1
        self.log_prob_w = np.zeros((n_topics, vocab_size))
        self.vocab_size = vocab_size
        self.num_topics = n_topics
        if initType=="seeded":
            self.corpus_initialize(corpus)
        if initType=="random":
            self.random_initialize(vocab_size,n_topics)
        else:
            self.zero_initialize(vocab_size,n_topics)

    def zero_initialize(self, vocab_size, n_topics):
        self.alpha = 0
        self.class_words = np.zeros((n_topics, vocab_size))
        self.class_total = np.zeros(n_topics)
        self.num_docs = 0.0

    def random_initialize(self, vocab_size, n_topics):
        self.alpha = 0
        self.class_words = np.full((n_topics, vocab_size), 1 / vocab_size + rnd.random())
        self.class_total = self.class_words.sum(axis=1)
        self.num_docs = 0.0

    def corpus_initialize(self, corpus):
        for k in self.num_topics:
            for i in xrange(self.NUM_INIT):
                d = round(rnd.random() * corpus.n_docs)
                print "Initialized with document {0}" % d
                doc = corpus.docs[d]
                for n in xrange(doc.length):
                    self.class_words[k, doc.words[n]] += doc.word_counts[n]

        self.class_words += 1
        self.class_total = self.class_words.sum(axis=1)

    def maximum_likelihood(self, estimate_alpha):
        for k in xrange(self.num_topics):
            for w in xrange(self.vocab_size):
                if self.class_words[k, w] > 0:
                    self.log_prob_w[k, w] = log(self.class_words[k, w]) - log(self.class_total[k])
                else:
                    self.log_prob_w[k, w] = -100

        if estimate_alpha == 1:
            self.alpha = opt_alpha(self.alpha, self.num_docs, self.num_topics)
            print "new alpha = {0} \n".format(self.alpha)

    def save(self,name):
        f_name=name+".beta"
        f=open(f_name)
        for i in xrange(self.num_topics):
            f.write(" ".join("{5:10f}\n".format(b) for b in self.log_prob_w[i]))
        f.close()
        f_name=name+".other"
        f=open(f_name)
        f.write("num topics: {0}\n".format(self.num_topics))
        f.write("vocab size: {0}\n".format(self.vocab_size))
        f.write("alpha: {5:10f}\n".format(self.alpha))
        f.close()








