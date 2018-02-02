#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:35:21 2017

@author: eti
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import time
import os
import cPickle
import matplotlib.pyplot as plt

from Load_attrnet_inputs import load_batch

from attnet import *
from params import *

def plotGraph( x1 , training_loss , valid_loss , title ) :
    plt.plot(x1, training_loss, label = "train")
    plt.plot(x1, valid_loss, label = "valid")
    plt.xlabel('number of epochs')
    plt.ylabel('Value' )
    plt.title(title)
    plt.legend()
    plt.figure()
    #plt.savefig(tit + '.png')   # save the figure to file
    plt.show()
    #plt.close()



def train(opt) :
    
    #loader = DataLoader(opt)
    #get num batches from loader
    num_batch = 2494
    
    model = Attnet(256,128,[100,100])
    model.cuda()

    infos = {}
    
    # Load valid data
    val_data = np.load('val_data_att.npy').item()
    tmp = [val_data['features'][0:5000] , val_data['object'][0:5000] , val_data['atts'][0:5000] ]
    tmp = [Variable(torch.from_numpy(t), requires_grad=False).cuda() for np.array(t)in tmp]
    vfc_feats, obj , atts  = tmp
    vlabels = [ obj , atts ]
    
    optimizer = optim.Adam(model.parameters(), lr= opt.learning_rate , weight_decay=opt.weight_decay)   
    
    train_loss = list()
    val_loss = list()
    
    for e in opt.epochs :   
       for b in num_batch :
       
        start = time.time()
        # Load data from train split (0)
        data = np.load('train_batch' + str(b) + '.npy' ).item()
        print('Read data:', time.time() - start)
        
        tmp = [data['features'], data['object'] , data['atts'] ]
        
        tmp = [Variable(torch.from_numpy(t), requires_grad=False).cuda() for np.array(t) in tmp]
        fc_feats, obj , atts = tmp
        labels = [ obj , atts ]
        
        optimizer.zero_grad()
        loss = loss_func(model(fc_feats) , labels)
        loss.backward()
        #utils.clip_gradient(optimizer, opt.grad_clip)
        torch.nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip)
        optimizer.step()
        train_loss.append(loss.data[0])
        torch.cuda.synchronize()
        end = time.time()
        print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
            .format(b, e, train_loss[-1], end - start))
       
        
       #validation
       loss = loss_func(model(vfc_feats) , vlabels)
       val_loss.append(loss.data[0])
       torch.cuda.synchronize()
       
       print "validation loss" + str(loss.data[0])
       
       current_score = - val_loss[-1]
       
       #checkpoints
       best_flag = False
       if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                #infos['batch'] = iteration
                infos['epoch'] = e
                #infos['iterators'] = loader.iterators
                #infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                #infos['vocab'] = loader.get_vocab()

                
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
       
                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)
       
       
    #plot the graphs
    x1 = list(range(1, epoch+1))
    title = 'Loss'
    plotGraph(x1,train_loss , val_loss , title)      

if __name__ == "__main__":
    
    opt = parse_opt()
    train(opt)