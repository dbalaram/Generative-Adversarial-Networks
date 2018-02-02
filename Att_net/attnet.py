#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:32:32 2017

@author: eti
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

#Declaring loss
loss_obj = nn.NLLLoss()
loss_attr = nn.BCELoss()


class Attnet(nn.module) :
    #multi-task classification for objects and attributes
    #takes image features as input
    
    def __init__(self, D_in , num_hid , num_classes ) :
        super(Attnet,self).__init__()
        self.num_layers = num_hid
        self.num_classes = num_classes
        self.D_in = D_in
        
    
        self.nn = nn.Sequential(
                  torch.nn.Linear(self.D_in, self.num_layers),
                  torch.nn.ReLU())
                  #torch.nn.Linear(self.num_layer[0] , self.num_layer[1]),
                  #torch.nn.ReLU())
                  #batch -norm
                  #torch.nn.BatchNorm1d(self.num_layer[1]))
    
        ##intialise word embeddings for layer 2
        #pretrained_weight =
        #self.nn[1].weight.data.copy_(torch.from_numpy(pretrained_weight))
        
        #for objects
        self.o = nn.Linear( self.num_layers, self.num_classes[0])
        self.ls = nn.LogSoftmax()
        #for attributes
        self.att = nn.Linear( self.num_layers , self.num_classes[1])
        self.sig = nn.Sigmoid()
    
    def forward(self,input) :
        
        r_out = self.nn(input)
        p_ob = self.ls(self.o(r_out))
        p_att = self.sig(self.att(r_out))
        
        return  [ p_ob , p_att ]
    
    
def loss_func( p , y) :

      #object loss -- categorical cross entropy
      l1 = loss_obj(p[0] , y[0])
      #attribute loss -- binary cross entropy
      l2 = loss_attr(p[1] , y[1])
      
      #return their sum      
      return l1 + l2