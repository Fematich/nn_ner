# -*- coding: utf8 -*-
"""
@author:    Matthias Feys (matthiasfeys@gmail.com)
@date:      %(date)
"""
import logging, os
import cPickle as pickle
import numpy as np

from config import trainingfile, testafile, testbfile, dataxyfile, numericannoationsfile, senna_numericannoationsfile, senna_dataxyfile
#from config import embdictfile
from config import senna_embdictfile, afeaturesfile

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger=logging.getLogger("annotationsDAO")
digitset=set(['0',',','.',' ','+','-'])
nerarray=['O','I-PER','I-ORG','I-LOC','I-MISC','B-PER','B-ORG','B-LOC','B-MISC','S-PER','S-ORG','S-LOC','S-MISC','E-PER','E-ORG','E-LOC','E-MISC']
nerdict=dict([(k,v) for v,k in enumerate(nerarray)])
inerarray=['O','I-PER','I-ORG','I-LOC','I-MISC','B-PER','B-ORG','B-LOC','B-MISC','I-PER','I-ORG','I-LOC','I-MISC','I-PER','I-ORG','I-LOC','I-MISC']
def transformannotationfile(annotationfile,embdictfile):
    annotations=[]
    embdict=pickle.load(open(embdictfile,'rb'))
    dstart=embdict['</s>']
    dunk=embdict['UNK']
    abuffer=[[dstart,0]]
    with open(annotationfile,'r') as fin:
        for line in fin:
            if line.startswith('-DOCSTART-'):
                continue
            if line=="\n":
                if len(abuffer)>1:
                    abuffer.append([dstart,0])
                    annotations.append(np.array(abuffer,dtype='int32'))
                abuffer=[[dstart,0]]
            else:
                parts=line.strip('\n').split()
                if parts[0] in embdict:
                    abuffer.append([embdict[parts[0]],nerdict[parts[3]]])
                else:
                    abuffer.append([dunk,nerdict[parts[3]]])
    return annotations
def get_nn_input(annotations,windowsize=5):
#    mtx=np.load(embmtxfile)
    x,y=[],[]
    tel=0
    for sentence in annotations:
        tel+=1
        if tel%140==0:
            logger.info("sentence progress %d "%(float(tel)/14041*100))
        for i in xrange(1,sentence.shape[0]-1):
            y.append(sentence[i][1])
#            x.append(np.hstack([mtx[sentence[max(min(j,sentence.shape[0]),0)][0]] for j in xrange(i-windowsize/2,i+windowsize/2+1)])) 
            x.append([sentence[max(min(j,sentence.shape[0]-1),0)][0] for j in xrange(i-windowsize/2,i+windowsize/2+1)])
    return np.array(x,dtype='int32'),np.array(y,dtype='int32')

def caps(word):
    uppers = [l.isupper() for l in word]
    if uppers[0]:
        if sum(uppers)==len(uppers):
            return 1#all upercase
        else:
            return 2#First capital letter only
    else:
        if sum(uppers)==0:
            return 0 #all lowercase
        else:
            return 3 #contains not initial uppercase
def senna_transformannotationfile(annotationfile):
    annotations=[]
    embdict=pickle.load(open(senna_embdictfile,'rb'))
    dstart=embdict['PADDING']
    dunk=embdict['UNKNOWN']
    abuffer=[[(dstart,0),0]]
    with open(annotationfile,'r') as fin, open('missing.txt','w') as missing:
        for line in fin:
            if line.startswith('-DOCSTART-'):
                continue
            if line=="\n":
                if len(abuffer)>1:
                    abuffer.append([[dstart,0],0])
                    annotations.append(abuffer)
                abuffer=[[[dstart,0],0]]
            else:
                parts=line.strip('\n').split()
                word = parts[0].lower()
                word=word.replace("1","0").replace("2","0").replace("3","0").replace("4","0").replace("5","0").replace("6","0").replace("7","0").replace("8","0").replace("9","0")
                
                if word.startswith('00-year'):
                    word=word[1:]
                elif word.startswith('0000-') or word.startswith('00-') or word.startswith('00/') or word.startswith('0000/'):
                    word="0/0/00"
                elif word.startswith("0:0") or word.startswith("00:0"):
                    word="0:0"
                elif word.endswith("0m"):
                    word="0m"
                elif word.endswith("0s"):
                    word="0s"
                numb=[(letter not in digitset) for letter in word]      
                if np.sum(numb)==0:
                    if word not in embdict:
                        if word.startswith('.'):
                            word=word[1:]
                        elif word.endswith(".0") or word.endswith(",0"):
                            word=word[:-2]
                        elif word.endswith(".00") or word.endswith(",00"):
                            word=word[:-3]
                        elif word.endswith(".000") or word.endswith(",000"):
                            word=word[:-4]
                        if word not in embdict:
                            word="0"
#                        word="00"
#                    print word, word.split('. , ')[0]
#                    wordparts=word.split(',')
#                    word=wordparts[0]
                if word in embdict:
                    abuffer.append([[embdict[word],caps(parts[0])],nerdict[parts[3]]])
                else:
                    missing.write(word+"\n")
                    abuffer.append([[dunk,caps(parts[0])],nerdict[parts[3]]])
    return annotations

def senna_get_nn_input(annotations,windowsize=5):
#    mtx=np.load(embmtxfile)
    x,y=[],[]
    tel=0
    for sentence in annotations:
        tel+=1
        if tel%140==0:
            logger.info("sentence progress %d "%(float(tel)/14041*100))
        for i in xrange(1,len(sentence)-1):
            y.append(sentence[i][1])
#            x.append(np.hstack([mtx[sentence[max(min(j,sentence.shape[0]),0)][0]] for j in xrange(i-windowsize/2,i+windowsize/2+1)])) 
            x.append([sentence[max(min(j,len(sentence)-1),0)][0] for j in xrange(i-windowsize/2,i+windowsize/2+1)])
    return np.array(x,dtype='int32'),np.array(y,dtype='int32')


def generate_output(modeldir,modelnumber=None):
    if modelnumber==None:
        digit=""
    else:
        digit=str(modelnumber)
    predictions=np.load(os.path.join(modeldir,"predictions%s.npy"%digit)).flatten()
#    train_set, valid_set, test_set = pickle.load(open(dataxyfile,'rb'))
#    predictions=test_set[1]
    with open(testbfile,'r') as fin, open(os.path.join(modeldir,"output%s.txt"%digit),'w') as fout:
        wordid=0
        for line in fin:
            if line.startswith('-DOCSTART-'):
                continue
            if line=="\n":
                fout.write(line)
            else:
#                print wordid
#                print predictions[wordid]
                fout.write(line.strip('\n')+" "+inerarray[predictions[wordid]]+"\n")
                wordid+=1

def summarize_output_errors(modeldir):
    with open(os.path.join(modeldir,"output.txt"),'r') as fin, open(os.path.join(modeldir,"errors.txt"),'w') as fout:
        for line in fin:
            if line=="\n":
                continue
            else:
                parts=line.strip("\n").split(" ")
                if parts[3]!=parts[4]:
                    fout.write(line)

if __name__ == '__main__':
    
##########################################    
########## WORD2VEC ######################
##########################################
#    train_annotations=transformannotationfile(trainingfile,embdictfile)
#    testa_annotations=transformannotationfile(testafile,embdictfile)
#    testb_annotations=transformannotationfile(testbfile,embdictfile)
#    logger.info("saving data")
#    np.save(numericannoationsfile,[train_annotations,testa_annotations,testb_annotations])
#
#    # generate NN input files    
#    annotations=np.load(numericannoationsfile)
#    data=[]
#    for annotationset in annotations:
#        x,y = get_nn_input(annotationset,5)
#        data.append([x,y])
#    pickle.dump(data,open(dataxyfile,'wb'))

##########################################    
############# SENNA ######################
##########################################
    train_annotations=senna_transformannotationfile(trainingfile)
    testa_annotations=senna_transformannotationfile(testafile)
    testb_annotations=senna_transformannotationfile(testbfile)
    logger.info("saving data")
    np.save(senna_numericannoationsfile,[train_annotations,testa_annotations,testb_annotations])

    # generate NN input files    
    annotations=np.load(senna_numericannoationsfile)
    data=[]
    for annotationset in annotations:
        x,y = senna_get_nn_input(annotationset,5)
        data.append([x,y])
    pickle.dump(data,open(senna_dataxyfile,'wb'))
    
#    afeatures=np.eye(4)
#    np.save(afeaturesfile,afeatures)

#    generate_output("/home/mfeys/work/data/ner/model500")

#    summarize_output_errors('/users/Mfeys/data/ner/model_h300_lr0.010000__39')