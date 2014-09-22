import numpy as np

class LdaSufficientStats(object):
    def __init__(self,n_topics,vocab_size,initType=None):
        self.alpha_ss = 0
        self.class_words = np.zeros((n_topics, vocab_size))
        self.class_total = np.zeros(n_topics)
        self.num_docs = 0
