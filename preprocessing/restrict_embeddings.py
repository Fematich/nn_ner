# -*- coding: utf8 -*-
"""
@author:    Matthias Feys (matthiasfeys@gmail.com)
@date:      %(date)
"""
import logging, os, itertools
import numpy as np
import cPickle as pickle

from config import senna_embmtxfile, senna_embdictfile, trainingfile, testafile,testbfile
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger=logging.getLogger("TODO")

embeddingsfile="news300.txt"
restrictedwordsfile="restrictedwords.txt"

sennadir="/home/mfeys/work/data/SENNA"
senna_embeddings=os.path.join(sennadir,"embeddings.txt")
senna_words=os.path.join(sennadir,"words.lst")

def restrict_embeddings(restrictedwords,embeddingsfile):
    embdict={}
    embmtx=[]
    with open(embeddingsfile,'r') as embeddings:
        lineno=0
        for line in embeddings:
            parts=line.strip('\n').split()
            if parts[0] in restrictedwords:
                embdict[parts[0]]=len(embdict)
                embmtx.append([float(val) for val in parts[1:]])
            lineno+=1
            if lineno%30000==0:
                logger.info("progress "+str(float(lineno)/3000000))
    return embdict,np.array(embmtx)
                
def transform_senna():
    embdict={}
    with open(senna_words,'r') as words:
        for line in words:
            embdict[line.strip('\n')]=len(embdict)
    embmtx=[]
    with open(senna_embeddings,'r') as embeddings:
        for line in embeddings:
            parts=line.strip('\n').split()
            embmtx.append([float(val) for val in parts])
    return embdict,np.array(embmtx)
   
def restrict_senna():
    rwords=set(['PADDING','UNKNOWN'])
    for doc in [trainingfile, testafile,testbfile]:
        with open(doc,'r') as document:
            for line in document:
                if not (line.startswith('-DOCSTART-') or line=="\n"):
                    parts=line.strip('\n').split()
                    word=parts[0].lower()
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
                    rwords.add(word)
    embdict={}
    embmtx=[]    
    with open(senna_words,'r') as words, open(senna_embeddings,'r') as embeddings:
        for w_line, e_line in itertools.izip(words,embeddings):
            word = w_line.strip('\n')
            if word in rwords:
                embdict[word]=len(embdict)
                parts=e_line.strip('\n').split()
                embmtx.append([float(val) for val in parts])
    return embdict,np.array(embmtx)
         
if __name__ == '__main__':
    ## word2vec
#    restwords=set([])
#    with open(restrictedwordsfile,'r') as restrictedwords:
#        for line in restrictedwords:
#            restwords.add(line.strip('\n'))
#    restwords.add('UNK')
#    restwords.add('</s>')
#    embdict,embmtx = restrict_embeddings(restwords,embeddingsfile)
#    logger.info("saving data")
#    np.save('embmtx',embmtx)
#    pickle.dump(embdict,open('embdict.pck','wb'))
    
    ## SENNA
    embdict,embmtx = restrict_senna()
    logger.info("saving data")
    np.save(senna_embmtxfile,embmtx)
    pickle.dump(embdict,open(senna_embdictfile,'wb'))