# -*- coding: utf8 -*-

import os

#datadir="/users/Mfeys/data/ner"
datadir="/home/mfeys/work/data/ner"
sennadir="/home/mfeys/work/data/SENNA"
storagedir="/home/mfeys/work/data/ner"

trainingfile=os.path.join(datadir,"eng.train")
testafile=os.path.join(datadir,"eng.testa")
testbfile=os.path.join(datadir,"eng.testb")
clean_trainingfile=os.path.join(datadir,"eng_clean.train")
clean_testafile=os.path.join(datadir,"eng_clean.testa")
clean_testbfile=os.path.join(datadir,"eng_clean.testb")

#embdictfile=os.path.join(datadir,"embdict.pck")
#embmtxfile=os.path.join(datadir,"embmtx.npy")
numericannoationsfile=os.path.join(datadir,"training.npy")
dataxyfile=os.path.join(datadir,"dataxy.pck")

senna_embmtxfile=os.path.join(sennadir,"senna_embmtx.npy")
senna_embdictfile=os.path.join(sennadir,"senna_embdict.pck")
senna_numericannoationsfile=os.path.join(datadir,"senna_training.npy")
senna_dataxyfile=os.path.join(datadir,"senna_dataxy.pck")
afeaturesfile=os.path.join(datadir,"caps.npy")