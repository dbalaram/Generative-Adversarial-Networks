#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 00:13:47 2017

@author: eti
"""

import os

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable



def prepare_discriminator_data( dis_data , gen , num_samples) :
    
    neg_captions = []
    neg_length = [] 
    #run forward pass on first 100 batches
    for i, (images, captions, lengths, objects_squares, os_length) in enumerate(dis_data[0:num_samples]):
           pcked_caption = gen.sample(images, captions, lengths, objects_squares, os_length)
           unpacked, unpacked_len =  torch.nn.utils.rnn.pad_packed_sequence(pcked_caption, batch_first=True)
           neg_captions.append(unpacked)
           neg_length.append(unpacked_len)
     
    #padd pos captions
    pos_captions = captions[num_samples:]
    pos_lengths = lengths[num_samples:]
    #pack_padded_sequence(captions[num_samples:], lengths[num_samples:], batch_first=True)[0]
    
    
    #pos neg captions    
    caps = torch.cat( (neg_captions , pos_captions) , 0)
    tlengths = torch.cat( (neg_length , pos_lengths) , 0)    
    # binary values 
    target = torch.ones(2 * num_samples)
    target[0:num_samples] = 0         
    

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    caps = caps[perm]
    images = images[perm]

    caps = Variable(caps).cuda()
    target = Variable(target).cuda()
    images = Variable(images).cuda()

    
    return   images , caps , target, tlengths   
   