#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 01:01:04 2017

@author: dhanashreebalaram
"""

from scipy.sparse import coo_matrix

import os

import numpy as np
from scipy.sparse import coo_matrix
import torch

from bounding_box_extractor import BBExtractor
m_obj = 100 # size of the vocabulary for objects
m_attr = 100 # size of the vocabulary for attributes
extractor = BBExtractor(max_num_objects=m_obj, max_num_attributes=m_attr) # use the same extractor for different batches
VG_ids = extractor.VG_ids_with_COCO

orig_image_batch_size = 100 # do not change here
features_path = '/home/ubuntu/preprocessing/resnet_features'

def load_batch(batch_id):
    slice_ = VG_ids[batch_id*orig_image_batch_size:(batch_id+1)*orig_image_batch_size]
    file_name = 'resnet_features_{:07}-{:07}.npz'.format(slice_[0], slice_[-1])
    load_path = os.path.join(features_path, file_name)
    loaded_data = np.load(load_path)
    features_batch = loaded_data['features']
    img_ids = loaded_data['img_ids']
    batch_data = extractor2.preprocess_images(slice_)
    attributes_label = batch_data.attributes_names
    objects_label = batch_data.object_names
    if (img_ids != batch_data.img_ids).any():
        print('!!!!!{}'.format(batch_id))
    return torch.FloatTensor(features_batch), attributes_label, objects_label, img_ids