# -*- coding: utf8 -*-
"""
@author:    Matthias Feys (matthiasfeys@gmail.com)
@date:      %(date)
"""
import logging
import numpy as np
import cPickle as pickle

from annotationsDAO import senna_transformannotationfile, senna_get_nn_input
from config import clean_trainingfile as trainingfile
from config import clean_testafile as testafile
from config import clean_testbfile as testbfile
from config import senna_numericannoationsfile,senna_dataxyfile

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger=logging.getLogger("TODO")

if __name__ == '__main__':
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