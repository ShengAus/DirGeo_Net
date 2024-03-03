import torch
import numpy as np
import random
import time
import os

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils.recorder import Recorder

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':         # SIngle sample cropped, no pre-loaded batches

    torch.cuda.empty_cache()
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    recorder = Recorder(opt)       # Recorder to monitor the progress
    total_iters = 0                # the total number of training iterations
    loss_list = []

    all_gt_means = []
    all_gt_stds = []

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        tot_loss = 0.
        counter = 0
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        model.set_current_epoch(epoch)
        # training
        for i, data in enumerate(dataset):
            data['dwi'] = data['dwi'].squeeze(0)
            data['gt_dti'] = data['gt_dti'].squeeze(0)
            data['wm_mask'] = data['wm_mask'].squeeze(0)
            data['grad'] = data['grad'].squeeze(0)
            # data['dwi_unnorm'] = data['dwi_unnorm'].squeeze(0)
            patch_nums =  len(data['dwi'])

            for j in range(0, patch_nums, opt.input_batch_sizes):
                total_iters += opt.input_batch_sizes
                epoch_iter += opt.input_batch_sizes

                inputs = {}
                inputs['dwi'] = data['dwi'][j:min(j+opt.input_batch_sizes, patch_nums),:,:,:,:]
                inputs['gt_dti'] = data['gt_dti'][j:min(j+opt.input_batch_sizes, patch_nums),:,:,:,:]
                inputs['wm_mask'] = data['wm_mask'][j:min(j+opt.input_batch_sizes, patch_nums),:,:,:,:]
                inputs['grad'] = data['grad']
                # inputs['dwi_unnorm'] = data['dwi_unnorm'][j:min(j+opt.input_batch_sizes, patch_nums),:,:,:,:]

                iter_start_time = time.time()  # timer for computation per iteration

                model.set_input(inputs)
                model.optimize_parameters()

                losses = model.get_current_losses()
                tot_loss += losses['l1']
                counter +=1
                
        recorder.print_epoch_losses(epoch, epoch_iter, {'l1' : tot_loss / counter}, time.time() - epoch_start_time)
        loss_list.append(tot_loss / counter)
        print('[TRAIN] loss: ', tot_loss / counter)

        if epoch % opt.display_freq == 0:
            recorder.plot_current_losses(epoch, {'l1' : tot_loss / counter})

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    
    print('End of training')
    recorder.close()
    
