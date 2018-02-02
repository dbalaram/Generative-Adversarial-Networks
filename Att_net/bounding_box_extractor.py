from collections import Counter, namedtuple
from io import BytesIO
from itertools import chain
import logging
import os
import json
import pickle

from IPython.display import display
import numpy as np
from PIL import Image
import requests
import torch

PreprocessedData = namedtuple('PreprocessedData',
                              ('object_squares',
                               'img_ids',
                               'original_sizes',
                               'object_names',
                               'attributes_names')
                             )

class WrongModeException(Exception):
    pass

class BBExtractor(object):
    def __init__(self, data_path='/home/ubuntu/data',
                 preprocessing_path='/home/ubuntu/preprocessing',
                 max_num_objects=100,
                 max_num_attributes=100,
                 use_sparse_matrices=True):
        self.data_path = data_path
        self.preprocessing_path = preprocessing_path
        self.image_data_path = os.path.join(data_path, 'image_data.json')
        self.objects_path = os.path.join(data_path, 'objects.json')
        self.attributes_path = os.path.join(data_path, 'attributes.json')
        self.img_dir = os.path.join(data_path, 'VG_100K_all')
        self.max_num_objects = max_num_objects
        self.max_num_attributes = max_num_attributes
        self.objects_info = self._objects_info()
        self.attributes_info = self._attributes_info()
        self.vocabulary = self._vocabulary()
        self.use_sparse_matrices = use_sparse_matrices
        
    def _objects_info(self):
        logger = logging.getLogger(__name__)
        with open(self.objects_path, 'r') as objects_file:
            objects_info_list = json.loads(objects_file.read())
        objects_info = {obj['image_id']: obj
                              for obj in objects_info_list}
        logger.info('Loaded objects info.')
        return objects_info
    
    def _attributes_info(self):
        logger = logging.getLogger(__name__)
        with open(self.attributes_path, 'r') as attributes_file:
            attributes_info_list = json.loads(attributes_file.read())
        attributes_info = {attr['image_id']: attr 
                                 for attr in attributes_info_list}
        logger.info('Loaded attributes info.')
        return attributes_info
        
    def _vocabulary(self):
        logger = logging.getLogger(__name__)

        vocabulary_path = os.path.join(self.preprocessing_path, 'vocabulary.pkl')
        if os.path.exists(vocabulary_path):
            with open(vocabulary_path, 'rb') as vocabulary_file:
                vocabulary = pickle.load(vocabulary_file)
            logger.info('Loaded vocabulary.')
        else:
            objects_counter = Counter()
            for objects_info_image in self.objects_info.values():
                for object_ in objects_info_image['objects']:
                    for name in object_['names']:
                        objects_counter[name] += 1
            sorted_obj_items = sorted(objects_counter.items(), key=lambda u: (-u[1], u[0]))
            vocabulary_objects = [item[0] for item in sorted_obj_items]
            obj_to_ix = {name: ix for ix, name in enumerate(vocabulary_objects)}

            attributes_counter = Counter()
            vocabulary_attributes = set()
            for attributes_info_image in self.attributes_info.values():
                for attributes_info_object in attributes_info_image['attributes']:
                    for attribute in attributes_info_object.get('attributes', []):
                        attributes_counter[attribute] += 1
            sorted_attr_items = sorted(attributes_counter.items(), key=lambda u: (-u[1], u[0]))
            vocabulary_attributes = [item[0] for item in sorted_attr_items]
            att_to_ix = {name: ix for ix, name in enumerate(vocabulary_attributes)}
            
            vocabulary = {
                'vocabulary_objects': vocabulary_objects,
                'obj_to_ix': obj_to_ix,
                'vocabulary_attributes': vocabulary_attributes,
                'att_to_ix': att_to_ix
            }
            with open(vocabulary_path, 'wb') as vocabulary_file:
                pickle.dump(vocabulary, vocabulary_file)
            logger.info('Generated vocabulary.')
        return vocabulary
    
    def get_img(self, img_id, url):
        img_name = '{}.jpg'.format(img_id)
        image_path = os.path.join(self.img_dir, img_name)
        if os.path.isfile(image_path):
            return Image.open(image_path)
        if url is not None:
            url_split = url.split('/')
            r = requests.get(url)
            r.raise_for_status()
            image_path = os.path.join(dir_path, url_split[-2], img_name)
            with open(image_path, 'wb') as image_file:
                image_file.write(r.content)
            return Image.open(BytesIO(r.content))
        raise requests.exceptions.HTTPError
        
    def label_vector(self, an, m):
        size = torch.Size((len(an), m))
        if self.use_sparse_matrices:
            indices_list = [[i, v]
                            for i, attr_row in enumerate(an)
                            for v in attr_row
                            if v < m]
            k = len(indices_list)
            if k > 0:
                indices = torch.LongTensor(list(zip(*indices_list)))
                values = torch.ones(k).byte()
                return torch.sparse.ByteTensor(indices,
                                               values,
                                               size)
            return torch.sparse.ByteTensor(size)
        else:
            R = torch.zeros(size).byte()
            for i, attr_row in enumerate(an):
                for v in attr_row:
                    if v < m:
                        R[i, v] = 1
            return R
        
    def preprocess_image(self, image_id, display_images=False):    
        objects_info_image = self.objects_info[image_id]
        attributes_info_image = self.attributes_info[image_id]
        img = self.get_img(img_id=objects_info_image['image_id'],
                           url=objects_info_image.get('image_url', None))

        if img.mode != 'RGB':
            raise WrongModeException

        if display_images:
            display(img)
            print('Original image\n')

        objects = objects_info_image['objects']
        N = len(objects)

        attributes = attributes_info_image['attributes']
        attributes_dict = {object_['object_id']:
                           object_.get('attributes', [])
                           for object_ in attributes}

        img_ids = np.ones(N, dtype=np.int) * image_id
        objects_squares = np.empty((N, 3, 224, 224), dtype=np.uint8)
        original_sizes = np.empty(N, dtype=np.int)
        objects_names = []
        attributes_names = []
        
        att_to_ix = self.vocabulary['att_to_ix']
        obj_to_ix = self.vocabulary['obj_to_ix']

        for i, (object_, attrs) in enumerate(zip(objects, attributes)):
            attrs = attributes_dict[object_['object_id']]
            attributes_names.append([att_to_ix[attr] for attr in attrs])
            original_sizes[i] = object_['w'] * object_['h']
            objs = object_['names']
            objects_names.append([obj_to_ix[obj] for obj in objs])

            # Image processing
            cropped = img.crop((object_['x'],
                                object_['y'],
                                object_['x'] + object_['w'],
                                object_['y'] + object_['h']))
            resized = cropped.resize((224, 224))

            if display_images:
                display(resized)
                print(' - '.join(object_['names']))
                print()

            objects_squares[i] = np.rollaxis(np.asarray(resized), axis=2, start=0)

        return PreprocessedData(objects_squares, img_ids, original_sizes, objects_names, attributes_names)
    
    def preprocess_images(self, image_ids):
        assert len(image_ids) >= 1
        logger = logging.getLogger(__name__)
        failed = []
        wrong_mode = []

        lists_per_image = []
        
        for _ in PreprocessedData._fields:
            lists_per_image.append([])

        for image_id in image_ids:
            try:
                result = self.preprocess_image(image_id)
            except requests.exceptions.HTTPError:
                failed.append(str(image_id))
                continue
            except WrongModeException:
                wrong_mode.append(str(image_id))
                continue
            except Exception as e:
                logger.error('Error with image {}.'.format(image_id))
                raise e

            for i, r in enumerate(result):
                lists_per_image[i].append(r)

        if failed:
            logger.warning('Failed to load image(s) {}.'.format(', '.join(failed)))

        if wrong_mode:
            logger.warning('Wrong mode for image(s) {}.'.format(', '.join(wrong_mode)))
        
        for i in range(0, 3):
            lists_per_image[i] = np.concatenate(lists_per_image[i])
            
        properties_objatt = ((3, self.max_num_attributes),
                             (4, self.max_num_objects))
        for i, max_num in properties_objatt:
            an = sum(lists_per_image[i], [])
            lists_per_image[i] = self.label_vector(an, max_num)
            
        return PreprocessedData(*lists_per_image)
    
    @property
    def VG_ids_with_COCO(self):
        if not hasattr(self, '_VG_ids_with_COCO'):
            image_data_path = os.path.join(self.data_path, 'image_data.json')
            with open(image_data_path, 'r') as D: #json file containing visual genome image information.
                F = json.loads(D.read()) # creates a list
                VG_ids = [] # list of common visual genome ids

                for f in F:
                    if f['coco_id'] is not None:
                        VG_ids.append(f['image_id'])
            self._VG_ids_with_COCO = sorted(VG_ids)
        return self._VG_ids_with_COCO