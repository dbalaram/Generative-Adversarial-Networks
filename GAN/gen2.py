#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:42:09 2017

@author: eti
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init


from model_encode_decode import * 

class Generator() :
    
    def __init__(self, embed_size , encoder_model_path , objs_attrs_size , \
                 hidden_size , vocab , num_layers , decoder_model_path , \
                 learning_rate, encoder_oa_model_path) : 
         
          self.embed_size = embed_size
          self.hidden_size = hidden_size
          self.num_layers =  num_layers
          self.objs_attrs_size = objs_attrs_size
          self.vocab = vocab
          
          #initialize encoder and decoder parameters
          self.encoder = EncoderCNN(embed_size, model_path=encoder_model_path)
          self.encoder_objs_attrs = EncoderObjsAttrsAverage(objs_attrs_size, 
                                                 embed_size,model_path=encoder_oa_model_path)
          self.decoder = DecoderRNN(embed_size, hidden_size, 
                         len(vocab), num_layers,
                         model_path=decoder_model_path)

          #optimizer
          params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) \
          + list(self.encoder.bn.parameters()) + list(self.encoder_objs_attrs.linear.parameters()) \
          + list(self.encoder_objs_attrs.bn.parameters()) 
          
          
          self.optimizer = torch.optim.Adam(params, lr=learning_rate)


    def forward(self , images , objects_squares , os_length , features , captions , lengths , objs_attrs_features) :

       #run forward pass on encoder and decoder
       self.decoder.zero_grad()
       self.encoder.zero_grad()
       self.encoder_objs_attrs.zero_grad()
       features = self.encoder(images)
       objs_attrs_features = self.encoder_objs_attrs(objects_squares, os_length)
       outputs = self.decoder(features, captions, lengths, objs_attrs_features)  
                
       return outputs
       
    def loss_mle(self , output , target):

         #loss function 
         criterion = nn.CrossEntropyLoss()  
         return criterion(output, target)
     
    def batchPGLoss(self, inp, target, reward , batch_size): 
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        loss = 0
        h = None
        for i in range(20):
            out, h = self.forward(inp[i], h)
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

       #loss for discriminator 
       #
       #loss = loss - log(rewards)   ( loss --> original generator loss)
        return loss/batch_size           
          
   
    def batchnewLoss(self, output , target , rewards , batch_size): 
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """
        loss = 0 
        criterion = nn.CrossEntropyLoss()  #generated caption vs true caption      
        for i in range(batch_size) :
              loss = loss + criterion(output[i], target[i]) - torch.log(rewards[i])   #( loss --> original generator loss)
        return loss/batch_size 
        
   
    
    
    def sample(self , images , captions, lengths, objects_squares, os_length) :  
        
        
        self.decoder.zero_grad()
        self.encoder.zero_grad()
        self.encoder_objs_attrs.zero_grad()
        features = self.encoder(images)
        objs_attrs_features = self.encoder_objs_attrs(objects_squares, os_length)
        outputs = self.decoder.sample(features, captions, lengths, objs_attrs_features) 
        
        return outputs
        
        