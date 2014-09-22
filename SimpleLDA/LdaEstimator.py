from LdaInference import LdaInference
from LdaModel import LdaModel
from scipy.special import psi
import numpy as np

def docEStep(doc,gamma,phi,model):
    likelihood=LdaInference.compute_likelihood(doc,model,phi,gamma)
    gamma_sum=0.0
    for k in xrange(model.num_topics):
        gamma_sum+=gamma[k]
        model.sufficient_statistics.alpha_ss += psi(gamma[k])

    model.sufficient_statistics.alpha_ss-=model.num_topics*psi(gamma_sum)

    for n in xrange(doc.length):
        for k in xrange(model.num_topics):
            model.sufficient_statistics.class_words[k][doc.words[n]]+=doc.word_counts[n]*phi[n,k]
            model.sufficient_statistics.class_total[k]+=doc.word_counts[n]*phi[n,k]

    model.sufficient_statistics.num_docs+=1
    return likelihood


def write_word_assignements(file,doc,phi,model):
    file.write("{:3d}".format(doc.length))
    for n in xrange(doc.length):
        file.write("{:4d}:{:2d}".format(doc.words[n],np.argmax(phi[n])))
    file.write("\n")


def save_gamma(fileName,gamma,num_docs,num_topics):
    f=open(fileName)
    for d in xrange(num_docs):
        line=" ".join("{5:10f}".format(g) for g in gamma[d])
        f.write(line)
        f.write("\n")
    f.close()


def run_EM(startFileName,directory,corpus,num_topics,startType=None):

    em_converged=10
    max_iter=10
    var_gamma=np.zeros((corpus.num_docs,corpus.num_docs))
    phi=np.zeros((corpus.max_doc_length,num_topics))
    if startType=="seeded":
        model=LdaModel(corpus.vocab_size,num_topics,"seeded",corpus)
    if startType=="random":
        model=LdaModel(corpus.vocab_size,num_topics,"random")
    else:
        model=LdaModel(corpus.vocab_size,num_topics)
    filename=directory+"/000"
    #model.save(filename)

    likelihood=0.0
    likelihood_old=0.0
    converged=1
    likelihood_filename=directory+"/likelihood.dat"
    likelihood_file=open(likelihood_filename,"wb")
    i=0
    while(converged <0 or converged > em_converged or (i<=2 and i<=max_iter)):
        i+=1
        print " ****** EM iteration {0} ***** \n".format(i)
        for d in xrange(corpus.num_docs):
            if d%1000==0:
                print "document {0}".format(d)
                likelihood+= docEStep(corpus.docs[d],var_gamma[d],phi,model)
        model.maximum_likelihood(1)
        converged=(likelihood_old-likelihood)/likelihood_old
        if converged<0:
            max_iter=max_iter*2
        likelihood_old=likelihood

        #likelihood_file.write("{:10.10f}".format(likelihood)+"\t"+"{:5.5e} \n".format(converged))
        if i%10==0:
            my_file=directory+"/{d}".format(i)
            #save lda model
            my_file=directory+"/{d}.gamma".format(i)
            save_gamma(my_file,var_gamma,corpus.n_docs,model.num_topics)

    #output model













