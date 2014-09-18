import numpy as np


class Corpus(object):
    def __init__(self):
        self.docs = []
        self.vocab_size = 0
        self.n_docs = 0


