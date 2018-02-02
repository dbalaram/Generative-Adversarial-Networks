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


class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=False, dropout=dropout)


    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h
    """
    def forward(self, input, img_ft , hidden):
        # input dim                                                # batch_size x seq_len
        emb = self.embeddings(input)                               # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
        out = F.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        ## do a dot product with image feature
        out = torch.dot(img_ft , out)        
        out = F.sigmoid(out)
        return out
    """
    def forward(self, inp, img_ft , input_lengths=None):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        """
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        ####
        emb = nn.utils.rnn.pack_padded_sequence(emb, input_lengths, batch_first=True)
        out, hidden = self.gru(emb, hidden)                     # 1 x batch_size x hidden_dim (out)
        out , _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        ####        
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        """
        #captions should be paddded
        
        emb = self.embeddings(captions)
        #emb = torch.cat((features.unsqueeze(1), embed), 1)
        packed = pack_padded_sequence(emb,  input_lengths , batch_first=True)  #time_step * batch  * embed_size
        _ , out = self.gru(packed) # op -->  t * batch * hidden_size , hid --> 1 *  batch * hidden_size 

        #out = self.linear(hiddens[0])          
        #out = F.tanh(out)
        ######
        out = out.view(-1, self.hidden_dim)  
        out = torch.dot(img_ft , out)        
        out = F.sigmoid(out)
        
        return out
    
      
    def batchClassify(self, inp , img_ft , lengths ):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len
            - img_ft : batch_size x feature_size
        Returns: out
            - out: batch_size ([0,1] score)
        """

        #h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, img_ft , lengths)
        return out.view(-1)

    def batchBCELoss(self, inp, tinp , img_ft  , lengths , target):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len 
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        
        #h = self.init_hidden(inp.size()[0])
        out1 = self.forward(inp,img_ft , lengths )
        out2 = self.forward(tinp,img_ft , lengths )
        #out3 = self.forward(tinp,img_ft , lengths )
        
        loss = -torch.log(out2) - torch.log(1 - out1 ) 
        return loss  #_fn(out, target)
