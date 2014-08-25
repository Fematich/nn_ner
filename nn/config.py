# -*- coding: utf8 -*-

import os

datadir="/home/mfeys/work/data"
#datadir="/users/Mfeys/data"

nerdir=os.path.join(datadir,"ner")
sennadir=os.path.join(datadir,"SENNA")
word2vecdir=os.path.join(datadir,"word2vec")

trainingfile=os.path.join(nerdir,"eng.train")
testafile=os.path.join(nerdir,"eng.testa")
testbfile=os.path.join(nerdir,"eng.testb")
#w2v_embdictfile=os.path.join(word2vecdir,"embdict.pck")
#w2v_embmtxfile=os.path.join(word2vecdir,"embmtx.npy")
numericannoationsfile=os.path.join(nerdir,"training.npy")
dataxyfile=os.path.join(nerdir,"dataxy.pck")

senna_embmtxfile=os.path.join(sennadir,"senna_embmtx.npy")
senna_embdictfile=os.path.join(sennadir,"senna_embdict.pck")
senna_numericannoationsfile=os.path.join(nerdir,"senna_training.npy")
senna_dataxyfile=os.path.join(nerdir,"senna_dataxy.pck")