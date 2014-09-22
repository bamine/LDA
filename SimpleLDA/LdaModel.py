import numpy as np
from scipy import log
import numpy.random as rnd
from LdaSufficientStats import LdaSufficientStats
from Utils import opt_alpha


class LdaModel(object):
    def __init__(self, vocab_size=0, n_topics=0,initType=None,corpus=None):
        self.alpha=10
        self.NUM_INIT = 1
        self.log_prob_w = np.zeros((n_topics, vocab_size))
        self.vocab_size = vocab_size
        self.num_topics = n_topics
        self.sufficient_statistics=LdaSufficientStats(n_topics,vocab_size)
        if initType=="seeded":
            self.corpus_initialize(corpus)
        if initType=="random":
            self.random_initialize(vocab_size,n_topics)
        else:
            self.zero_initialize(vocab_size,n_topics)

    def zero_initialize(self, vocab_size, n_topics):
        self.sufficient_statistics.alpha_ss = 0
        self.sufficient_statistics.class_words = np.zeros((n_topics, vocab_size))
        self.sufficient_statistics.class_total = np.zeros(n_topics)
        self.sufficient_statistics.num_docs = 0

    def random_initialize(self, vocab_size, n_topics):
        self.sufficient_statistics.alpha = 0
        self.sufficient_statistics.class_words = np.full((n_topics, vocab_size), 1 / vocab_size + rnd.random())
        self.sufficient_statistics.class_total = self.class_words.sum(axis=1)
        self.sufficient_statistics.num_docs = 0.0

    def corpus_initialize(self, corpus):
        for k in xrange(self.num_topics):
            for i in xrange(self.NUM_INIT):
                d = round(rnd.random() * corpus.n_docs)
                print "Initialized with document {0}" % d
                doc = corpus.docs[d]
                for n in xrange(doc.length):
                    self.sufficient_statistics.class_words[k, doc.words[n]] += doc.word_counts[n]

        self.sufficient_statistics.class_words += 1
        self.sufficient_statistics.class_total = self.sufficient_statistics.class_words.sum(axis=1)

    def maximum_likelihood(self, estimate_alpha):
        for k in xrange(self.num_topics):
            for w in xrange(self.vocab_size):
                if self.sufficient_statistics.class_words[k, w] > 0:
                    self.log_prob_w[k, w] = log(self.sufficient_statistics.class_words[k, w]) - log(self.class_total[k])
                else:
                    self.log_prob_w[k, w] = -100

        if estimate_alpha == 1:
            self.alpha = opt_alpha(self.sufficient_statistics.alpha_ss, self.sufficient_statistics.num_docs, self.num_topics)
            print "new alpha = {0} \n".format(self.alpha)

    def save(self,name):
        f_name=name+".beta"
        f=open(f_name,"wb")
        for i in xrange(self.num_topics):
            f.write(" ".join("{5:10f}\n".format(b) for b in self.log_prob_w[i]))
        f.close()
        f_name=name+".other"
        f=open(f_name,"wb")
        f.write("num topics: {0}\n".format(self.num_topics))
        f.write("vocab size: {0}\n".format(self.vocab_size))
        f.write("alpha: {5:10f}\n".format(self.alpha))
        f.close()

    def load(self,name):
        other_name=name+".other"
        beta_name=name+".beta"
        print "loading "+other_name
        f=open(other_name)
        for line in f:
            num_topics, vocab_size, alpha=line.split()
            self.num_topics=int(num_topics)
            self.vocab_size=int(vocab_size)
            self.alpha=float(alpha)
        f.close()
        self.log_prob_w=np.loadtxt(beta_name,delimiter=" ")










