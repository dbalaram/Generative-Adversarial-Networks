#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:29:23 2017

@author: eti
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb
from dis_enc_dec import * 


class Discriminator(nn.Module):

    def __init__(self, embed_size , encoder_model_path , hidden_size ,\
                 vocab , num_layers , decoder_model_path , \
                 learning_rate) : 
         
          self.embed_size = embed_size
          self.hidden_size = hidden_size
          self.num_layers =  num_layers
          #self.objs_attrs_size = objs_attrs_size
          self.vocab = vocab
          
          #initialize encoder and decoder parameters
          self.img = EncoderCNN(embed_size, model_path=encoder_model_path)
          self.lang = DecoderRNN(embed_size, hidden_size, 
                         len(vocab), num_layers,
                         model_path=decoder_model_path)

          #optimizer
          params = list(self.decoder.parameters()) + list(self.encoder.linear.parameters()) \
          + list(self.encoder.bn.parameters())
        
        
          self.optimizer = torch.optim.Adam(params, lr=learning_rate)           
        

    def forward(self , images , captions , lengths) :

       #run forward pass on encoder and decoder
       self.lang.zero_grad()
       self.img.zero_grad()
       features = self.img(images)
       out = self.lang(captions , lengths)     
       
       ######  
       out = torch.dot( features , out)        
       out = F.sigmoid(out)
        
       return out
    
    
    def batchClassify(self, images , captions , lengths) :
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len
            - img_ft : batch_size x feature_size
        Returns: out
            - out: batch_size ([0,1] score)
        """

       #run forward pass on encoder and decoder
       self.lang.zero_grad()
       self.img.zero_grad()
       features = self.img(images)
       out = self.lang(captions , lengths)     
       
       ######  
       out = torch.dot( features , out)        
       out = F.sigmoid(out)
        
       return out

   
  
    def batchBCELoss(self, out , target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """
 
        loss_fn = nn.BCELoss()
        return loss_fn(out, target)
 
    def newloss( self , true_sig , gen_sig) :
        
        loss = - torch.log(true_sig) - torch.log(1 - gen_sig) 
        return loss