import numpy as np
from scipy import log
from scipy.special import gammaln, psi,polygamma
import scipy.optimize


def logSum(loga, logb):
    if loga < logb:
        return logb + log(1 + np.exp(loga - logb))
    else:
        return loga + log(1 + np.exp(logb - loga))

def alpha_likelihood(a,ss,D,K):
    return D*(gammaln(K*a)-K*gammaln(a)) + (a-1)*ss

def d_alpha_likelihood(a,ss,D,K):
    return D*(K*psi(K*a)-K*psi(a)) + ss

def d2_alpha_likelihood(a,ss,D,K):
    return D*(K*K*polygamma(1,K*a)-K*polygamma(1,K*a)(a))


def opt_alpha(alpha_ss,num_docs,num_topics):
    def f(a):
        return alpha_likelihood(a,alpha_ss,num_docs,num_topics)

    def fprime(a):
        return d_alpha_likelihood(a,alpha_ss,num_docs,num_topics)

    log_alpha_opt=scipy.optimize.fmin_bfgs(f,10,fprime)
    return np.exp(log_alpha_opt)




