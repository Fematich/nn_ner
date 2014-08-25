# -*- coding: utf8 -*-

"""
@author:    Matthias Feys (matthiasfeys@gmail.com)
@date:      %(date)
"""
import logging, os, sys
import numpy as np
from config import testbfile

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger=logging.getLogger("TODO")

nerarray=['O','I-PER','I-ORG','I-LOC','I-MISC','B-PER','B-ORG','B-LOC','B-MISC','S-PER','S-ORG','S-LOC','S-MISC','E-PER','E-ORG','E-LOC','E-MISC']

def generate_output(modeldir,modelnumber=None, predictions=None):
    if modelnumber==None:
        digit=""
    else:
        digit=str(modelnumber)
    if predictions==None:
        predictions=np.load(os.path.join(modeldir,"predictions%s.npy"%digit))
    predictions=predictions.flatten()
    firstWord=True
#    train_set, valid_set, test_set = pickle.load(open(dataxyfile,'rb'))
#    predictions=test_set[1]
    with open(testbfile,'r') as fin, open(os.path.join(modeldir,"output%s.txt"%digit),'w') as fout:
        wordid=0
        for line in fin:
            if line.startswith('-DOCSTART-'):
                firstWord=True
                continue
            if line=="\n":
                firstWord=True
                fout.write(line)
            else:
#                print wordid
#                print predictions[wordid]
                prediction=nerarray[predictions[wordid]].split('-')
                if prediction[0]=="B" or prediction[0]=="S":
                    if not firstWord:
                        if prediction[1]==nerarray[predictions[wordid-1]].split('-')[-1]:
                            prediction[0]="B"
                        else:
                            prediction[0]="I"
                    else:
                        prediction[0]="I"
                else:
                    if prediction[0]!="O":
                        prediction[0]="I"
                fout.write(line.strip('\n')+" "+'-'.join(prediction)+"\n")
                wordid+=1
                firstWord=False
                
if __name__ == '__main__':
    if len(sys.argv)>2:
        generate_output(sys.argv[1],modelnumber=sys.argv[2])
    else:
        generate_output(sys.argv[1])