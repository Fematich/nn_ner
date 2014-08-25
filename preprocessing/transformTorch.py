# -*- coding: utf8 -*-
"""
@author:    Matthias Feys (matthiasfeys@gmail.com)
@date:      %(date)
"""
import logging
import numpy as np

from config import senna_dataxyfile as trainingfile
from config import testbfile

nerarray=['O','I-PER','I-ORG','I-LOC','I-MISC','B-PER','B-ORG','B-LOC','B-MISC','S-PER','S-ORG','S-LOC','S-MISC','E-PER','E-ORG','E-LOC','E-MISC']

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger=logging.getLogger("TODO")


def generate_output(filename):
    predictions=[]    
    with open(filename,"r") as fin:
        for line in fin:
            predictions.append(int(line.strip("\n"))-1)
    prev_prediction_type=""
#    train_set, valid_set, test_set = pickle.load(open(dataxyfile,'rb'))
#    predictions=test_set[1]
    with open(testbfile,'r') as fin, open("/home/mfeys/work/data/ner_torch/output.txt",'w') as fout:
        wordid=0
        for line in fin:
            try:
                if line.startswith('-DOCSTART-'):
                    prev_prediction_type=""
                    continue
                if line=="\n":
                    fout.write(line)
                    prev_prediction_type=""
                else:
    #                print wordid
    #                print predictions[wordid]
                    prediction=nerarray[predictions[wordid]].split('-')
                    if prediction[0]=="B" or prediction[0]=="S":
                        if prediction[1]==prev_prediction_type:
                            prediction[0]="B"
                        else:
                            prediction[0]="I"
                    else:
                        if prediction[0]!='O':
                            prediction[0]="I"
                    fout.write(line.strip('\n')+" "+'-'.join(prediction)+"\n")
                    prev_prediction_type=prediction[-1]
                    wordid+=1
            except Exception:
                logger.info("not all test-examples tested")

if __name__ == '__main__':
    generate_output("/home/mfeys/work/data/ner_torch/ner_results_embedding.txt")
#    train_set, valid_set, test_set = np.load(trainingfile)
#    
#    with open('/home/mfeys/work/data/ner_torch/ner_training.txt', 'w') as fout:
#        for i in xrange(train_set[0].shape[0]):
##            fout.write(str(train_set[1][i]+1)+" "+" ".join([str(train_set[0][i][j][0]+1)+" "+str(train_set[0][i][j][1]+1) for j in xrange(5)])+"\n")
#            fout.write(" ".join([str(train_set[0][i][j][0]+1) for j in xrange(5)])+" "+" ".join([str(train_set[0][i][j][1]+1) for j in xrange(5)])+" "+str(train_set[1][i]+1)+"\n")
#    with open('/home/mfeys/work/data/ner_torch/ner_testing.txt', 'w') as fout:
#        for i in xrange(test_set[0].shape[0]):
##            fout.write(str(test_set[1][i]+1)+" "+" ".join([str(test_set[0][i][j][0]+1)+" "+str(test_set[0][i][j][1]+1) for j in xrange(5)])+"\n")
#
#
#            fout.write(" ".join([str(test_set[0][i][j][0]+1) for j in xrange(5)])+" "+" ".join([str(test_set[0][i][j][1]+1) for j in xrange(5)])+" "+str(test_set[1][i]+1)+"\n")
