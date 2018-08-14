#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:26:55 2017

@author: firojalam
"""

import numpy as np

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, model, train_history):
        current_valid = train_history['valid_loss'][-1]
        current_epoch = train_history['epoch'][-1]
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = model.store_params()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
#            nn.load_params_from(self.best_weights)


patience = 100
best_valid = np.inf
best_valid_epoch = 0
best_weights = None
train_history={}
train_history['valid_loss']={}
train_history['epoch']={}

current_valid = train_history['valid_loss'][-1]
current_epoch = train_history['epoch'][-1]
if current_valid < best_valid:
    best_valid = current_valid
    best_valid_epoch = current_epoch
    best_weights = m.store_params()
elif best_valid_epoch + patience < current_epoch:
    print("Early stopping.")
    print("Best valid loss was {:.6f} at epoch {}.".format(
        best_valid, best_valid_epoch))
