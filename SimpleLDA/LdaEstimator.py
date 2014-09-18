from LdaInference import LdaInference
from scipy.special import psi
import numpy as np


class LdaEstimator(object):
    def docEStep(self,doc,gamma,phi,model):
        likelihood=LdaInference.compute_likelihood(doc,model,phi,gamma)
        gamma_sum=0.0
        for k in xrange(model.num_topics):
            gamma_sum+=gamma[k]
            model.alpha += psi(gamma[k])

        model.alpha-=model.num_topics*psi(gamma_sum)

        for n in doc.length:
            for k in model.num_topics:
                model.class_words[k][doc.words[n]]+=doc.word_counts[n]*phi[n,k]
                model.class_total[k]+=doc.word_counts[n]*phi[n,k]

        model.num_docs+=1
        return likelihood

    def write_word_assignements(self,file,doc,phi,model):
        file.write("{:3d}",doc.length)
        for n in xrange(doc.length):
            file.write("{:4d}:{:2d}",doc.words[n],np.argmax(phi[n]))
        file.write("\n")

    def save_gamma(self):






