#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 23:15:14 2017

@author: eti
"""

import os
import pickle

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO

# from preprocessing.features_getter import CNNFeaturesGetter
# from preprocessing.attnet_nobatchnorm import Attnet

# class AttnetFeaturesGetter():
#     def __init__(self, model_path='/home/ubuntu/preprocessing/model-best.pth'):
#         self.model = Attnet(256, 128, [100, 100])
#         self.model.load_state_dict(torch.load(model_path))
#         self.model.cuda()
#         self.model.eval()    
        
#     def __call__(self, features):
#         fc_feats = Variable(torch.from_numpy(np.array(features)), requires_grad=False).cuda()
#         return self.model.nn.forward(fc_feats)


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, json_instances, transform=None,
                 val_ids_path='/home/ubuntu/preprocessing/val.npy',
                 test_ids_path='/home/ubuntu/preprocessing/test.npy',
                 mode='train'):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.coco_ins = COCO(json_instances)
        if mode == 'train':
            val_ids = set(np.load(val_ids_path))
            test_ids = set(np.load(test_ids_path))
            train_ids = set(self.coco.anns.keys()) - val_ids - test_ids
            self.ids = sorted(train_ids)
        elif mode == 'valid':
            self.ids = sorted(np.load(val_ids_path))
        elif mode == 'test':
            self.ids = sorted(np.load(test_ids_path))
        self.vocab = vocab
        self.transform = transform
#         self.cnn_features_getter = CNNFeaturesGetter()
#         self.attnet_feature_getter = AttnetFeaturesGetter()
        
    def get_object_squares(self, img_id, img):
        ins_ids = self.coco_ins.getAnnIds(imgIds=img_id)
        ins = self.coco_ins.loadAnns(ins_ids)

        objects_squares = np.empty((len(ins), 3, 224, 224), dtype=np.uint8)
        for i, instance in enumerate(ins):
            x, y, width, height = instance['bbox']
            cropped = img.crop((x, y, x+width, y+height))
            resized = cropped.resize((224, 224))
            objects_squares[i] = np.rollaxis(np.asarray(resized), axis=2, start=0)

        return objects_squares.astype(float)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco

        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        
        # Objects/attrs
        folder_file = self.coco_ins.loadImgs(img_id)[0]['coco_url'].rsplit('/', 2)[-2:]
        img_path = os.path.join(self.root, '..', *folder_file)
        img = Image.open(img_path).convert('RGB')
        objects_squares = self.get_object_squares(img_id, img)
#         features = self.cnn_features_getter(objects_squares)
#         objs_attrs = self.attnet_feature_getter(features)

        if objects_squares.shape[0] == 0:
            objects_squares = None
        else:
            objects_squares = torch.FloatTensor(objects_squares)
        
        return image, target, objects_squares

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, objs_attrs).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption, objs_attrs). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, objects_squares = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]      
        
    objects_squares_lengths = [os.size(0) if os is not None else 0
                               for os in objects_squares] 
    
    objects_squares = torch.cat([os for os in objects_squares
                                 if os is not None], 0)
    return images, targets, lengths, objects_squares, objects_squares_lengths


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers, json_instances, mode='train'):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform,
                       json_instances=json_instances,
                       mode=mode)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader