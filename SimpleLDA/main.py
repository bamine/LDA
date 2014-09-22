from Corpus import Corpus
import LdaEstimator

def main():
    corpus=Corpus("ap.dat")
    LdaEstimator.run_EM("","result",corpus,10)

if __name__=="__main__":
    main()
