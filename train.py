# -*- coding: utf8 -*-
"""
@author:    Matthias Feys (matthiasfeys@gmail.com)
@date:      %(date)
"""
from nn.trainer import Trainer

if __name__ == '__main__':
    trainer=Trainer(batch_size=32)
    trainer.train_model(lr_scheme="constant",initial_learning_rate=0.01,lr_global=False,n_epochs=1000)
