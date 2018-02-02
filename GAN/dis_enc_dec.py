#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 21:03:39 2017

@author: eti
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:29:02 2017

@author: eti
"""

import os

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from preprocessing.features_getter import CNNFeaturesGetter
from preprocessing.attnet import Attnet


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, model_path=None):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights(model_path)
        
    def init_weights(self, model_path):
        """Initialize the weights."""
        if model_path is not None:
            self.load_state_dict(torch.load(model_path))
        else:
            self.linear.weight.data.normal_(0.0, 0.02)
            self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
    
class DecoderRNN(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, model_path=None):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights(model_path)
    
    def init_weights(self, model_path):
        """Initialize weights."""
        if model_path is not None:
            self.load_state_dict(torch.load(model_path))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)
            self.linear.weight.data.uniform_(-0.1, 0.1)
            self.linear.bias.data.fill_(0)
        
    def forward(self, captions , lengths , initial_hidden_state=None):
        """Decode image feature vectors and generates captions."""
        embeddings = []
        embeddings.append(self.embed(captions))
        embeddings = torch.cat(embeddings, 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hidden , out  = self.gru(packed, initial_hidden_state)
        #outputs = self.linear(hiddens)
        #####################send the last hidden state
        return out
    
