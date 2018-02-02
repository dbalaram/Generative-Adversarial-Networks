#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 02:40:28 2017

@author: eti
"""

from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn

from generator import * 
from  discriminator import * 
from  helper import * 
from params import *

CUDA = True
#VOCAB_SIZE = 5000
MAX_SEQ_LEN = 20
#START_LETTER = 0
BATCH_SIZE = 64
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 50
#POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

def train_generator_MLE(gen, epochs , args ) : #oracle, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    
    ##intialise everything
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             json_instances=args.instances_path,
                             shuffle=True, num_workers=args.num_workers,
                             mode='train') 

    
    if torch.cuda.is_available():
        gen.encoder.cuda()
        gen.encoder_objs_attrs.cuda()
        gen.decoder.cuda()

    best_val_score = None
    
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0.0
        for i, (images, captions, lengths, objects_squares, os_length) in enumerate(data_loader):
            
                 # Set mini-batch dataset
                 images = to_var(images, volatile=True)
                 captions = to_var(captions)
                 targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                 #run forward pass
                 outputs =   gen.forward(images, captions, lengths, objects_squares, os_length )
                 
                 loss = gen.loss_mle(outputs,targets)
                 loss.backward()
                 gen.optimizer.step()
                 
                 total_loss += loss.data[0]

                if (i / 64)  == 10 : #% ceil(
            #                ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
           #     print('.', end='')
                       sys.stdout.flush()
                       
                       
        # Save the models
        if (epoch+1) % args.save_step == 0:
                torch.save(gen.decoder.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_MLE_decoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(gen.encoder.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_MLE_encoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(gen.encoder_objs_attrs.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_MLE_encoder_objs_attrs-%d-%d.pkl' %(epoch+1, i+1)))               
                       
                       
                       
        mle_train_loss.append((total_loss / (i+1)))

        
        #validation
        
        vdata = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform , 5000 , #args.batch_size,
                             json_instances=args.instances_path,
                             shuffle=True, num_workers=args.num_workers,
                             mode='valid') 
        images, captions, lengths, objects_squares, os_length = vdata[0] 

        # Set mini-batch dataset
        images = to_var(images, volatile=True)
        captions = to_var(captions)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        #run forward pass
        outputs =   gen.sample(images , objects_squares , os_length \
                                         , features , captions , lengths , objs_attrs_features)
                 
        vloss = gen.loss_mle(outputs,targets)       
        
        mle_valid_loss.append(vloss.data[0])
        
        #save losses
        np.save('mle_train_loss.npy' , mle_train_loss)
        np.save('mle_valid_loss.npy' , mle_valid_loss)
        
        
        
        #save_best_models
        if best_val_score is None or vloss < best_val_score:
                    best_val_score = vloss
                    torch.save(gen.decoder.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_MLE_decoder-best.pkl'))
                    torch.save(gen.encoder.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_MLE_encoder-best.pkl'))
                    torch.save(gen.encoder_objs_attrs.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_MLE_encoder_objs_attrs-best.pkl')) 
        
        print(' average_train_NLL = %.4f'  % (mle_train_loss[-1]) , 
              ' average_valid_NLL = %.4f'  % (mle_valid_loss[-1])) #oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))


def train_generator_PG(gen, epochs , args , dis ) :
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    tloss =  0.0
    
    for i, (images, captions, lengths, objects_squares, os_length) in enumerate(data_loader): 
           # Set mini-batch dataset
           images = to_var(images, volatile=True)
           captions = to_var(captions)
           targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

           #run forward pass
           outputs =   gen.forward(images, captions, lengths, objects_squares, os_length)
            
           #rewards
           rewards = dis.batchClassify(images , captions , lengths )
           loss = gen.batchPGLoss(outputs,targets, args.batch_size)
           loss.backward()
           gen.optimizer.step()
                 
           tloss += loss.data[0]
           
           if (i / 64 )  == 10 : #% ceil(
            #                ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
           #     print('.', end='')
                       sys.stdout.flush()       
        
    tloss = tloss / ( i +  1 )
    #run_validation
    vdata = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform , 5000 , #args.batch_size,
                             json_instances=args.instances_path,
                             shuffle=True, num_workers=args.num_workers,
                             mode='valid') 
    
    images, captions, lengths, objects_squares, os_length = vdata[0] 

    # Set mini-batch dataset
    images = to_var(images, volatile=True)
    captions = to_var(captions)
    targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
      
    outputs =  gen.sample(images, captions, lengths, objects_squares, os_length)
    rewards = dis.batchClassify( vtarget)
    vloss = gen.loss_pg(outputs,targets)
    

    return tloss, vloss




def train_discriminator(discriminator, gen , args , d_steps, epochs ):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
  
    #sample some validation data
    vdis_data = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             json_instances=args.instances_path,
                             shuffle=True, num_workers=args.num_workers,
                             mode='valid')[0:100] 
    
    vdis_imgs , vdis_captions , vdis_targets , vdis_lengths =  \
           helper.prepare_discriminator_data( vdis_data , gen  , 100 )   
    
    for d_step in range(d_steps):

         num_samples = 200
         # Build data loader
         dis_data = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             json_instances=args.instances_path,
                             shuffle=True, num_workers=args.num_workers,
                             mode='train')[0:2*num_samples] 
        
         #create a new dataset 
         dis_imgs , dis_captions , dis_targets  , dis_lengths =  helper.prepare_discriminator_data( dis_data , gen  , num_samples )        
        
        
         for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0.0
            total_acc = 0
             
            for i in xrange(num_samples) :
                start = i*args.batch_size
                end = start + args.batch_size
                images, captions, targets , lengths = dis_imgs[start:end] , dis_captions[start:end] \
                                       , dis_targets[start:end] , dis_lengths[start:end]
                
                # Set mini-batch dataset
                images = to_var(images, volatile=True)
                captions = to_var(captions)
                #targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                
                out = discriminator.forward(images , captions , lengths)
                #gen_sig = 
                loss = discriminator.batchBCELoss( out , targets ) #loss_fn(true_sig , gen_sig)
                loss.backward()
                discriminator.optimizer.step()

                total_loss += loss.data[0]

            total_loss = total_loss / num_samples 
            #do validation
            #vdis_imgs , vdis_captions , vdis_targets
            # Set mini-batch dataset
            images = to_var(vdis_imgs, volatile=True)
            captions = to_var(vdis_captions) 
            out = discriminator.forward(images , captions  , lengths)
            #gen_sig = 
            vloss = discriminator.batchBCELoss( out , vdis_targets )

         return total_loss, vloss
       
# MAIN
if __name__ == '__main__':
    
    args = parser.parse_args()
    
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
       
    # Load the pretrained models
    encoder_model_path = os.path.join(args.model_path, 'encoder-{}'.format(suffix))
    #encoder_oa_model_path = os.path.join(args.model_path, 'encoder_objs_attrs-{}'.format(suffix))
    decoder_model_path = os.path.join(args.model_path, 'decoder-{}'.format(suffix))    
        
        
    gen = Generator(args.embed_size , encoder_model_path , args.objs_attrs_size , \
                 args.hidden_size , args.vocab , args.num_layers , decoder_model_path , \
                 args.learning_rate,encoder_oa_model_path=None)
    dis = Discriminator(args.embed_size , encoder_model_path , 
                 args.hidden_size , args.vocab , args.num_layers , decoder_model_path , \
                 args.learning_rate)



    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    #gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)    
    train_generator_MLE(gen, MLE_TRAIN_EPOCHS)


    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    _ , _  = train_discriminator(dis , gen, args , 50, 3 )

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    #oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
    #                                           start_letter=START_LETTER, gpu=CUDA)
    #print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)


    genpg_valid_loss = list()
    genpg_train_loss = list()
    dispg_valid_loss = list()
    dispg_train_loss = list()
    
    
    ####################### Adversial Network ##########################################
    ##intialise everything
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             json_instances=args.instances_path,
                             shuffle=True, num_workers=args.num_workers,
                             mode='train') 

    
    if torch.cuda.is_available():
        gen.encoder.cuda()
        gen.encoder_objs_attrs.cuda()
        gen.decoder.cuda()
    
    best_val_score = None
    dis_val_score = None
    
    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()        
        
     
        tloss , vloss = train_generator_PG( gen, data_loader , dis )
        
        genpg_train_loss.append(tloss)
        genpg_valid_loss.append(vloss)
        
            
        # Save the models
        if (epoch+1) % args.save_step == 0:
                torch.save(gen.decoder.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_PG_decoder-%d.pkl' %(epoch+1)))
                torch.save(gen.encoder.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_PG_encoder-%d.pkl' %(epoch+1)))
                torch.save(gen.encoder_objs_attrs.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_PG_encoder_objs_attrs-%d.pkl' %(epoch+1)) 
         
        #save_best_models
        if best_val_score is None or vloss < best_val_score:
                    best_val_score = vloss
                    torch.save(gen.decoder.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_PG_decoder-best.pkl' ))
                    torch.save(gen.encoder.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_PG_encoder-best.pkl'))
                    torch.save(gen.encoder_objs_attrs.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_PG_encoder_objs_attrs-best.pkl'))    
        

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        dtloss , dvloss = train_discriminator(dis, dis_optimizer,gen, oracle, 5, 3 , tot_batches , gen)
        
        dispg_train_loss.append(dtloss)
        dispg_valid_loss.append(dvloss)
        
            
        # Save the models
        if (epoch+1) % args.save_step == 0:
                torch.save(gen.decoder.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_PG_decoder-%d.pkl' %(epoch+1)))
                torch.save(gen.encoder.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_PG_encoder-%d.pkl' %(epoch+1)))
                torch.save(gen.encoder_objs_attrs.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Generator_PG_encoder_objs_attrs-%d.pkl' %(epoch+1)) 
         
        #save_best_models
        if dis_val_score is None or dvloss < dis_val_score:
                    best_val_score = vloss
                    torch.save(dis.lang.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Discriminator_PG_decoder-best.pkl' ))
                    torch.save(dis.img.state_dict(), 
                           os.path.join(args.save_model_path, 
                                        'Discriminator_PG_encoder-best.pkl'))    
        
np.save('genpg_train_loss.npy' , genpg_train_loss)
np.save('genpg_valid_loss.npy' , genpg_valid_loss)


np.save('dispg_train_loss.npy' , dispg_train_loss)
np.save('dispg_valid_loss.npy' , dispg_valid_loss)
        