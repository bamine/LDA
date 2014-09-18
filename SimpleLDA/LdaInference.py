import math
import numpy as np
from scipy.special import psi, gamma
from scipy import log


def log_sum(log_a, log_b):
    if log_a < log_b:
        v = log_b + log(1 + np.exp(log_a - log_b))
    else:
        v = log_a + log(1 + np.exp(log_b - log_a))
    return v


class LdaInference(object):
    @staticmethod
    def compute_likelihood(doc, model, phi, var_gamma):
        var_gamma_sum = 0
        dig = np.zeros(model.n_topics)
        for k in xrange(model.n_topics):
            dig[k] = psi(var_gamma[k])
            var_gamma_sum += var_gamma[k]
        digsum = psi(var_gamma_sum)
        likelihood = gamma(model.alpha * model.num_topics) - model.num_topics * gamma(model.alpha) - gamma(
            var_gamma_sum)

        for k in xrange(model.num_topics):
            likelihood += (model.alpha - 1) * (dig[k] - digsum) + gamma(var_gamma[k]) - (var_gamma[k] - 1) * (
                dig[k] - digsum)

        for n in xrange(doc.length):
            if phi[n, k] > 0:
                likelihood += doc.word_counts[n] * (
                    phi[n, k] * ((dig[k] - digsum) - log(phi[n, k]) + model.log_prob_w[k, doc.words[n]]))

        return likelihood

    def run_inference(self, doc, model, var_gamma, phi):
        var_converged = 10
        var_max_iter = 10

        converged = 1
        likelihood = 0
        likelihood_old = 0
        oldphi = np.zeros(model.num_topics)
        digamma_gam = np.zeros(model.num_topics)

        for k in xrange(model.num_topics):
            var_gamma[k] = model.alpha + doc.total / model.num_topics
            digamma_gam = psi(var_gamma[k])
            for n in xrange(doc.length):
                phi[n, k] = 1 / model.num_topics

        var_iter = 0
        while (converged > var_converged) and (var_iter < var_max_iter or var_max_iter == -1):
            var_iter += 1
            for n in xrange(doc.length):
                phisum = 0
                for k in xrange(model.num_topics):
                    oldphi[n, k] = phi[n, k]
                    phi[n, k] = digamma_gam[k] + model.log_prob_w[k, doc.words[n]]
                    if k > 0:
                        phisum = log_sum(phisum, phi[n, k])
                    else:
                        phisum = phi[n, k]

                for k in xrange(model.num_topics):
                    phi[n, k] = np.exp(phi[n, k] - phisum)
                    var_gamma[k] += doc.word_counts[n] * (phi[n, k] - oldphi[k])
                    digamma_gam[k] = psi(var_gamma[k])

                likelihood = self.compute_likelihood(doc, model, phi, var_gamma)
                assert not math.isnan(likelihood)
                converged = (likelihood_old - likelihood) / likelihood_old
                likelihood_old = likelihood

                print "[LDA INFERENCE] likelihood: {:5} - error: {:3f} \n".format(likelihood, converged)

        return likelihood










