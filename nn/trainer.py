# -*- coding: utf8 -*-
"""
@author:    Matthias Feys (matthiasfeys@gmail.com)
@date:      %(date)
"""

import logging, os, time
import numpy as np
import cPickle as pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

import theano
import theano.tensor as T

from config import senna_dataxyfile as trainingfile
from config import senna_embmtxfile, nerdir, testbfile
from network import SennaNER
from util import hardtanh

logger=logging.getLogger("senna_retrain_ner")
nerarray=['O','I-PER','I-ORG','I-LOC','I-MISC','B-PER','B-ORG','B-LOC','B-MISC','S-PER','S-ORG','S-LOC','S-MISC','E-PER','E-ORG','E-LOC','E-MISC']
class Trainer():
    def __init__(self,batch_size=16, seed=1234,nhu=300,width=5,n_out=len(nerarray),activation_f="hardtanh",
                 embeddingfile=senna_embmtxfile,trainingfile=trainingfile,paramfile=None):
        modeldir=os.path.join(nerdir,"models",'model_%i'%(len(os.listdir(nerdir+"/models"))))
        os.mkdir(modeldir)   
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=os.path.join(modeldir,'log.txt'), level=logging.INFO, 
                            format='%(asctime)s : %(levelname)s : %(message)s')    
        logger.info("\n"+"\n".join(["\t%s : "%key+str(val) for key,val in locals().iteritems() if key!="self"]))    
        self.modeldir=modeldir
        self.batch_size = batch_size
        activation=None        
        if activation_f=="hardtanh":
            activation=hardtanh
        elif activation_f=="tanh":
            activation=T.tanh
        self.load_data(embeddingfile,trainingfile,batch_size)
        #==============================================================================
        #         BUILD MODEL
        #==============================================================================
        logger.info('... building the model')
        # allocate symbolic variables for the data
        self.index = T.iscalar()  # index to a [mini]batch
        self.x = T.itensor3('x')  # the data is presented as matrix of integers
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        self.permutation = T.ivector('permutation')
        if paramfile!=None:
            params=pickle.load(open(paramfile,"rb"))
        else:
            params=None
        self.model = SennaNER(input=self.x, embeddings=self.embeddings,features=capsfeatures,n_out=n_out, mini_batch_size=batch_size,
                                       nhu=nhu,width=width,activation=activation,seed=seed,params=params)

        self.test_model = theano.function(inputs=[self.index],
                outputs=self.model.errors(self.y),
                givens={
                    self.x: self.test_set_x[self.index * batch_size:(self.index + 1) * batch_size],
                    self.y: self.test_set_y[self.index * batch_size:(self.index + 1) * batch_size]},
                name="test_model")
    
        self.validation_cost = theano.function(inputs=[self.index],
                outputs=self.model.negative_log_likelihood(self.y),
                givens={
                    self.x: self.valid_set_x[self.index * batch_size:(self.index + 1) * batch_size],
                    self.y: self.valid_set_y[self.index * batch_size:(self.index + 1) * batch_size]},
                name="validation_cost")
    
        self.predictions = theano.function(inputs=[self.index],
                outputs=self.model.predictions,
                givens={
                    self.x: self.test_set_x[self.index * batch_size:(self.index + 1) * batch_size]},
                name="predictions")
    
        self.visualize_hidden = theano.function(inputs=[self.index],
                outputs=self.model.HiddenLayer.output,
                givens={
                    self.x: self.valid_set_x[self.index * batch_size:(self.index + 1) * batch_size]},
                name="visualize_hidden")

    def load_data(self,embeddingsfile,dataset,batch_size):
        logger.info('... loading data')        
        self.embeddings=np.load(embeddingsfile)
        rng=np.random.RandomState(1234)
        self.capsfeatures=np.asarray(rng.uniform(
                    low=-np.sqrt(1. / (1)),
                    high=np.sqrt(1. / (1)),
#                    low=-np.sqrt(3. / 1),
#                    high=np.sqrt(3. / 1),
                    size=(4,5)), dtype=theano.config.floatX)
        train_set, valid_set, test_set = np.load(dataset)
        self.n_train_batches = train_set[0].shape[0] / batch_size
        self.n_valid_batches = valid_set[0].shape[0] / batch_size
        self.n_test_batches = test_set[0].shape[0] / batch_size
    
        self.train_set_size = train_set[0].shape[0]
        
        def shared_dataset(data_xy, borrow=True):
            data_x, data_y = data_xy
            shared_x = theano.shared(np.asarray(data_x,
                                                   dtype='int32'),
                                     borrow=borrow)
            shared_y = theano.shared(np.asarray(data_y,
                                                   dtype='int32'),#dtype=theano.config.floatX),
                                     borrow=borrow)
            return shared_x, shared_y#T.cast(shared_y, 'int32')

        self.test_set_x, self.test_set_y = shared_dataset(test_set)
        self.valid_set_x, self.valid_set_y = shared_dataset(valid_set)
        self.train_set_x, self.train_set_y = shared_dataset(train_set)    
        
    def train_model(self, lr_scheme,initial_learning_rate=0.01, min_lr=0.00001,learning_rate_decay=0.05,constant_steps=None,L1_reg=0.0000, L2_reg=0.0000,lr_global=False, n_epochs=100,momentum_term=0.9):
        logger.info("\n"+"\n".join(["\t%s : "%key+str(locals()[key]) for key in ["lr_scheme","lr_global","min_lr","initial_learning_rate","learning_rate_decay","L1_reg","L2_reg","n_epochs"]]))    
        cost = self.model.negative_log_likelihood(self.y) \
         + L2_reg * self.model.L2 #\
#         + L1_reg * self.model.L1 
        
        self.learning_rate = theano.shared(np.float32(initial_learning_rate))
        if constant_steps==None:
            self.constant_steps = np.inf
        else:
            self.constant_steps = constant_steps
        self.lr_scheme = lr_scheme

        def gen_updates_sgd():
            gparams = [theano.grad(cost, param) for param in self.model.params]
            updates = []
            for param_i, grad_i, n_in in zip(self.model.params, gparams, self.model.n_ins):
                if "embeddings" not in str(param_i):
                    updates.append((param_i, param_i - self.learning_rate/n_in * grad_i))
                else:
                    updates.append((param_i, param_i - self.learning_rate * grad_i))
            return updates
          
        def gen_updates_sgd_global():
            gparams = [theano.grad(cost, param) for param in self.model.params]
            updates = []
            for param_i, grad_i in zip(self.model.params, gparams):
                updates.append((param_i, param_i - self.learning_rate * grad_i))
            return updates

#        def gen_updates_regular_momentum(loss, all_parameters, learning_rate, momentum, weight_decay):
#            all_grads = [theano.grad(loss, param) for param in all_parameters]
#            updates = []
#            for param_i, grad_i in zip(all_parameters, all_grads):
#                mparam_i = theano.shared(param_i.get_value()*0.)
#                v = momentum * mparam_i - weight_decay * learning_rate * param_i  - learning_rate * grad_i
#                updates.append((mparam_i, v))
#                updates.append((param_i, param_i + v))
#            return updates
#        
#        def gen_updates_own_momentum():
#            agparams=[theano.shared(value=np.zeros(p.get_value().shape, dtype=theano.config.floatX), name='ag_'+p.name) \
#                for p in self.model.params]   # averaged gradients
#            gparams = [] # gradients
#            for pid,param in enumerate(self.model.params):
#                gparam = T.grad(cost, param)
#                gparams.append(gparam)
#            updates = []
#            for param, gparam, agparam, n_in in zip(self.model.params, gparams, agparams, self.model.n_ins):
#                updates.append((agparam,np.float32(1-momentum_term)*agparam + np.float32(momentum_term)*gparam))            
#                if lr_global:
#                    updates.append((param, param - self.learning_rate/n_in * (np.float32(1-momentum_term)*agparam + np.float32(momentum_term)*gparam)))
#                else:
#                    updates.append((param, param - self.learning_rate * (np.float32(1-momentum_term)*agparam + np.float32(momentum_term)*gparam)))
#            return updates
        if lr_global:
            updates = gen_updates_sgd_global()
        else:
            updates = gen_updates_sgd()
        train_model = theano.function(inputs=[self.index,self.permutation], outputs=theano.Out(cost, borrow=True),
            updates=updates,
            givens={
                self.x: self.train_set_x[self.permutation[self.index * self.batch_size:(self.index + 1) * self.batch_size]],
                self.y: self.train_set_y[self.permutation[self.index * self.batch_size:(self.index + 1) * self.batch_size]]},
            name="train_model")

        #==============================================================================
        # train model
        #==============================================================================
        theano.printing.pydotprint(train_model)
        logger.info('... training')

        min_valid_cost = np.inf
        best_epoch = 0
        test_score = 0.
        start_time = time.clock()
    
        epoch = 0
        self.trainingscosts=[]
        self.validationcosts=[]
        training_costs=[10]
        while (epoch <= n_epochs):
            self.trainingscosts.append(np.mean(training_costs))
            validation_costs = [self.validation_cost(i) for i
                                 in xrange(self.n_valid_batches)]
            self.validationcosts.append(np.mean(validation_costs))
            self.monitor_update()
            if self.validationcosts[-1]<min_valid_cost:
                min_valid_cost=self.validationcosts[-1]
                best_epoch=epoch
                self.test_error(epoch)
            if epoch%25==0:
                pickle.dump(self.model,open(os.path.join(self.modeldir,'model%i.pck'%epoch),'wb'),protocol=pickle.HIGHEST_PROTOCOL)
                hidden_values = [self.visualize_hidden(i) for i
                                 in np.random.randint(0,self.n_valid_batches,30)]
                image = np.vstack(hidden_values)
                binary_image = (image>0.999) | (image<-0.999)
                plt.imshow(binary_image,cmap=plt.cm.get_cmap('gray'), interpolation='nearest')
                plt.savefig(os.path.join(self.modeldir,'binary_hidden%i.png'%epoch))
                plt.clf()
                test_predictions = [self.predictions(i) for i
                                       in xrange(self.n_test_batches)]
                np.save(os.path.join(self.modeldir,"predictions.npy"),test_predictions)
                generate_output(self.modeldir,modelnumber=epoch, predictions=np.array(test_predictions))
            training_costs=[]
            perm=np.random.permutation(self.train_set_size).astype(np.int32)
            for minibatch_index in xrange(self.n_train_batches):
                training_costs.append(train_model(minibatch_index,perm))
            
            if epoch>0:
                if self.lr_scheme!="constant":
                    if self.lr_scheme=="continuous" and epoch>self.constant_steps:
                        self.learning_rate.set_value(np.float32(initial_learning_rate*(1+learning_rate_decay* self.constant_steps)/(1+learning_rate_decay*max(epoch,self.constant_steps))))
                    elif ((self.validationcosts[-1]-self.validationcosts[-2])>0 and (self.validationcosts[-1]-np.min(self.validationcosts))>0.01 and \
                    np.argmin(self.validationcosts)!=(len(self.validationcosts)-2)) or \
                    (((self.trainingscosts[-1]-self.trainingscosts[-2])>0) and (np.argmin(self.trainingscosts)!=(len(self.trainingscosts)-2))):
                        if self.lr_scheme=="stepwise":
                            self.learning_rate.set_value(np.float32(self.learning_rate.get_value()/3))
                        elif self.lr_scheme=="continuous":
                            self.constant_steps=epoch-1
                            self.learning_rate.set_value(np.float32(initial_learning_rate*(1+learning_rate_decay*self.constant_steps)/(1+learning_rate_decay*max(epoch,self.constant_steps))))
                    if self.learning_rate.get_value()<min_lr:
                        self.learning_rate.set_value(np.float32(min_lr))
                        self.lr_scheme=="constant" 
            epoch = epoch + 1
        end_time = time.clock()
        logger.info(('Optimization complete. Best validation score of %f %% '
               'obtained at epoch %i, with test performance %f %%') %
              (min_valid_cost, best_epoch, test_score * 100.))
        logger.info('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        self.monitor_update()
        test_predictions = [self.predictions(i) for i
                                       in xrange(self.n_test_batches)]
        generate_output(self.modeldir,predictions=np.array(test_predictions))
#        np.save(os.path.join(self.modeldir,"predictions.npy"),test_predictions)
    def monitor_update(self):
        plt.plot(self.trainingscosts,label='training')
        plt.plot(self.validationcosts,label='validation')
        plt.legend()
        plt.savefig(os.path.join(self.modeldir,'costs.png'))
        plt.clf()       
        logger.info('epoch %i,validation cost %f, training cost %f, lr %f' %
             (len(self.trainingscosts)-1, self.validationcosts[-1],self.trainingscosts[-1],self.learning_rate.get_value()))
    def test_error(self,epoch):
        test_losses = [self.test_model(i) for i
                               in xrange(self.n_test_batches)]
        test_score = np.mean(test_losses)
        logger.info(('     epoch %i, test error of best model %f %%') %
                      (epoch, test_score * 100.))

def generate_output(modeldir,modelnumber=None, predictions=None):
    if modelnumber==None:
        digit=""
    else:
        digit=str(modelnumber)
    if predictions==None:
        predictions=np.load(os.path.join(modeldir,"predictions%s.npy"%digit))
    predictions=predictions.flatten()
    prev_prediction_type=""
#    train_set, valid_set, test_set = pickle.load(open(dataxyfile,'rb'))
#    predictions=test_set[1]
    with open(testbfile,'r') as fin, open(os.path.join(modeldir,"output%s.txt"%digit),'w') as fout:
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
    trainer=Trainer(batch_size=32)
    trainer.train_model(lr_scheme="constant",initial_learning_rate=0.01,lr_global=False,n_epochs=1000)