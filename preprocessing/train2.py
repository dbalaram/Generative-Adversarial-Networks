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
import pickle
import matplotlib.pyplot as plt

# from Load_attrnet_inputs import load_batch

from attnet2 import *
#from params import *

def plotGraph( x1 , training_loss , valid_loss , title ) :
    plt.plot(x1, training_loss, label = "train")
    plt.plot(x1, valid_loss, label = "valid")
    plt.xlabel('number of epochs')
    plt.ylabel('Value' )
    plt.title(title)
    plt.legend()
    #plt.figure()
    plt.savefig(title + '.png')   # save the figure to file
    #plt.show()
    plt.close()



def train() :
    
    #loader = DataLoader(opt)
    #get num batches from loader
    num_batch = 753
    checkpoint_path = '/home/ubuntu/preprocessing'
    model = Attnet(256,128,[ 100 ,100 ])
    model.cuda()
    epochs = 1000
    infos = {}
    best_val_score = None
    # Load valid data
    val_data = np.load('val_data_att.npy').item()
    tmp = [val_data['features'], val_data['object'] , val_data['atts'] ]
    #print type(val_data['atts']) , type(val_data['object']) , type(val_data['atts'][0]) 
    
        
    #tmp = [Variable(torch.from_numpy(t), requires_grad=False).cuda() for np.array(t) in tmp]
    vfc_feats = Variable(torch.from_numpy(np.array(tmp[0])), requires_grad=False).cuda()
    #obj = Variable(torch.from_numpy(np.array(tmp[1])), requires_grad=False).cuda() 
    atts  =  Variable(torch.from_numpy(np.array(tmp[2])), requires_grad=False).cuda()
    #atts = obj 
    vlabels = atts #[ obj , atts ]
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001 ) #, weight_decay=0.001)   
    
    train_loss = list()
    val_loss = list()
    
    for e in range(epochs) :   
       tl = 0  
       for b in range(num_batch) :
       
           start = time.time()
           # Load data from train split (0)
           data = np.load('train_batch' + str(b) + '.npy' ).item()
           print('Read data:', time.time() - start)
        
           tmp = [data['features'], data['object'] , data['atts'] ]
        
           fc_feats = Variable(torch.from_numpy(np.array(tmp[0])), requires_grad=False).cuda()
           #obj = Variable(torch.LongTensor(torch.from_numpy(np.array(tmp[1]))), requires_grad=False).cuda() 
           #print np.array(tmp[2]).shape
           atts  =  Variable(torch.from_numpy(np.array(tmp[2])), requires_grad=False).cuda()
           #tmp = [Variable(torch.from_numpy(t), requires_grad=False).cuda() for np.array(t) in tmp]
           #fc_feats, obj , atts = tmp
           #atts = obj
           #print atts.size()
           labels = atts #[ obj , atts ]
        
           optimizer.zero_grad()
           loss = loss_func(model(fc_feats) , labels)
           loss.backward()
        #utils.clip_gradient(optimizer, opt.grad_clip)
           torch.nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip)
           optimizer.step()
           tl = tl + loss.data[0] 
           torch.cuda.synchronize()
           end = time.time()
           print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                  .format(b, e, loss.data[0], end - start))
       
       train_loss.append((tl / (b+1) )) 
        
       #validation
       loss = loss_func(model(vfc_feats) , vlabels)
       val_loss.append(loss.data[0])
       torch.cuda.synchronize()
       
       print ("validation loss" , str(loss.data[0]))
       
       current_score = - val_loss[-1]
       
       #checkpoints
       best_flag = False
       if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                model_path = os.path.join( checkpoint_path ,'model_onlyatt.pth')
                torch.save(model.state_dict() , model_path)
                print("model saved to {}".format(model_path))
                optimizer_path = os.path.join(checkpoint_path , 'optimizer2.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                #infos['batch'] = iteration
                infos['epoch'] = e
                #infos['iterators'] = loader.iterators
                #infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                #infos['vocab'] = loader.get_vocab()

                
                with open(os.path.join(checkpoint_path , 'infos2_'+ str(e) +'.pkl'), 'wb') as f:
                    pickle.dump(infos, f)
       
                if best_flag:
                    model_path = os.path.join(checkpoint_path , 'model-best_onlyatt.pth')
                    torch.save(model.state_dict() , model_path)
                    print("model saved to {}".format(model_path))
                    with open(os.path.join(checkpoint_path , 'infos2_'+ str(e) +'-best.pkl'), 'wb') as f:
                        pickle.dump(infos, f)
       
    #save both the losses 
    np.save('trainloss_att.npy' , train_loss) 
    np.save('validloss_att.npy' , val_loss)
          
    #plot the graphs
    x1 = list(range(1, epochs+1))
    title = 'Loss-onlyatt'
    plotGraph(x1,train_loss , val_loss , title)    
    
    
def test(tdata) :
    
#laod pretrained model  
# Load valid data
    #val_data = np.load('val_data_att.npy').item()
    tmp = tdata['features']  # tdata['object'] , tdata['atts'] ]
    fc_feats = Variable(torch.from_numpy(np.array(tmp)), requires_grad=False).cuda()       
    
    #model = models.setup(opt)
    model = Attnet(256,128,[ 100 ,100 ])
    model.load_state_dict(torch.load('model-best_onlyatt.pth'))
    model.cuda()
    model.eval()    

    att_features = model.nn.forward(fc_feats)   
    
    return att_features
    
    

if __name__ == "__main__":
    #train = 0
    #opt = parse_opt()
    #if train : 
    train() #opt)
    #else :
    #     test(1) 