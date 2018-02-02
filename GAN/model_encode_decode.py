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

class AttnetFeaturesGetter():
    def __init__(self, model_path='/home/ubuntu/preprocessing/model-best.pth'):
        self.model = Attnet(256, 128, [100, 100])
        self.model.load_state_dict(torch.load(model_path))
        self.model.cuda()
        self.model.eval()    
        
    def __call__(self, features):
        return self.model.nn.forward(features)


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
   

class EncoderObjsAttrs(object):
    def __init__(self):
        self.cnn_features_getter = CNNFeaturesGetter()
        self.attnet_feature_getter = AttnetFeaturesGetter()
        
    def objs_attrs(self, objects_squares):
        features = self.cnn_features_getter(objects_squares)
        objs_attrs = self.attnet_feature_getter(features)
        return objs_attrs.data


class EncoderObjsAttrsAverage(nn.Module, EncoderObjsAttrs):
    def __init__(self, objs_attrs_size, embed_size, model_path=None):
        super(EncoderObjsAttrsAverage, self).__init__()
        EncoderObjsAttrs.__init__(self)
        self.objs_attrs_size = objs_attrs_size
        self.linear = nn.Linear(objs_attrs_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights(model_path)
        
    def init_weights(self, model_path):
        """Initialize the weights."""
        if model_path is not None:
            self.load_state_dict(torch.load(model_path))
        else:
            assert False
            self.linear.weight.data.normal_(0.0, 0.02)
            self.linear.bias.data.fill_(0)
        
    def forward(self, objects_squares, lengths):
        objs_attrs = self.objs_attrs(objects_squares)
        averages = torch.FloatTensor(len(lengths), self.objs_attrs_size)
        if torch.cuda.is_available():
            averages = averages.cuda()
        index = 0
        # TODO: Is there a built-in function for that?
        for i, length in enumerate(lengths):
            if length > 0:
                next_index = index + length
                averages[i] = objs_attrs[index:next_index].mean(dim=0)
                index = next_index
            else:
                averages[i] = 0
        averages = Variable(averages)
        features = self.bn(self.linear(averages))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, model_path=None):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
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
        
    def forward(self, features, captions, lengths, objs_attrs_features, initial_hidden_state=None):
        """Decode image feature vectors and generates captions."""
        embeddings = []
        if objs_attrs_features is not None:
            embeddings.append(objs_attrs_features.unsqueeze(1))
        embeddings.append(features.unsqueeze(1))
        embeddings.append(self.embed(captions))
        embeddings = torch.cat(embeddings, 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed, initial_hidden_state)[0]
        outputs = self.linear(hiddens)
        return outputs
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()
