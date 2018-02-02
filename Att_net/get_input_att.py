#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 22:48:59 2017

@author: eti
"""

#splitting data into train , split , valid

#saves dicitonaryies with keys :  features , object , atts -->  index values
import numpy as np
import torch
from collections import defaultdict

import data_loader as name
import torch.nn as nn
# name.load_data()


print "blah"
batch_size = 64
num_batches = 517
count = 0
val_id = np.load('val.npy')
test_id = np.load('test.npy')

val_data = defaultdict(list)
train_batch = defaultdict(list)
test_data = defaultdict(list)
print "start fetching data"
for b in range(num_batches) : 
    ft , obj , att , vids = name.load_batch(b)
    print " batch loaded "
    vids = np.array(vids)
    #get indexes of common test data
    ind = np.nonzero(np.in1d(vids, val_id))[0]  
    vids = np.delete(vids, ind)
    print " train data ---" , len(val_data)
    
    test_data['features'].append(ft[ind,:]) 
    for i in ind :   
        #get index of ones
        if np.any(obj[i,:]==1) :
            obj_l = np.where(obj[i,:]==1)
        if np.any(att[i,:]==1) :   
            att_l = np.where(att[i,:]==1)           
           
      
        test_data['object'].append(obj_l)    
        test_data['atts'].append(att[i,:])
        
    #get indexes of common valid data
    ind = np.nonzero(np.in1d(vids, val_id))[0]
    vids = np.delete(vids, ind)
    print " valid data ---" , len(val_data)
    
    val_data['features'].extend(ft[ind,:]) 
    
    for i in ind :   
        #get index of ones
        if np.any(obj[i,:]==1) :
            obj_l = np.where(obj[i,:]==1)
        if np.any(att[i,:]==1) :   
            att_l = np.where(att[i,:]==1)           
        vids = np.delete(vids, ind)    
           
        val_data['object'].append(obj_l)    
        val_data['atts'].append(att[i,:])
        
    #get  indexes of train data
    for i in vids :
        if np.any(obj[i,:]==1) :
            obj_l = np.where(obj[i,:]==1)
        if np.any(att[i,:]==1) :   
            att_l = np.where(att[i,:]==1)           
        vids = np.delete(vids, ind)    
        train_batch['features'].append(ft[i,:])    
        train_batch['object'].append(obj_l)    
        train_batch['atts'].append(att[i,:])
        if len(train_batch) == batch_size :
             #save_this_batch
             np.save('train_batch' + str(count) + '.npy', train_batch)
             print " train file saved "
             train_batch = defaultdict(list)
             count = count + 1
             
#save test and val array
np.save('val_data_att.npy', val_data)
np.save('test_data_att.npy' , test_data)             
#loading file
#np.load('my_file.npy').item()             