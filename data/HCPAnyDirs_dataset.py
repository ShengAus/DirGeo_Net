import os
import torch
import nibabel as nib
import numpy as np
import pickle
from data.base_dataset import BaseDataset
from data_loading.HCPAnyDirs_sample_loader import HCPAnyDirsSampleLoader
from processing.Nonorm_processing import NonormProcessing

# dataset can load dwis of anydirections
# dwis (G, W, H, D)
# 

class HCPAnyDirsDataset(BaseDataset):   

    def __init__(self, opt):
        """
        Initialize this dataset class.
        """
        BaseDataset.__init__(self, opt)

        hcp_split_path = os.path.join(self.root, 'HCP_list_split_80_20.pickle')
        if not os.path.exists(hcp_split_path):
            raise IOError(
                "hcp splited list path, {}, could not be resolved".format(hcp_split_path)
            )
            exit(0)
        
        with open(hcp_split_path, 'rb') as handle:
            sub_list = pickle.load(handle)
            if opt.isTrain:
                self.sample_list = sub_list['train']
            else:
                self.sample_list = sub_list['test']

        # ## debugging uncomment it
        # self.sample_list = [self.sample_list[0]]
        # self.sample_list = self.sample_list[0:20]

        self.sample_loader = HCPAnyDirsSampleLoader(root = self.root)
        self.padded_dirs = opt.padded_dirs
        self.processing = NonormProcessing(norm = opt.data_norm, bounding = opt.bounding
                        , crop = True, isTrain = opt.isTrain, patch_shape = opt.patch_shape , patch_overlap = opt.patch_overlap)


    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """

        return parser


    def __len__(self):
        """Return the total number of images in the dataset."""

        return len(self.sample_list)


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        uni = self.sample_list[index]
        # debug uncomment it
        # uni = '111312'

        grad_dirs = np.random.randint(6, self.padded_dirs+1)
     
        sample = self.sample_loader.load_sample(uni, grad_dirs)
        self.processing.preprocessing(sample)
        # debug uncomment it
        # print(sample.gt_dti_mean)
     
        grad = torch.from_numpy(np.concatenate(([[0.,0.,0.]], sample.grad[...,0:-1]), axis = 0)) # (7, 3)
        
        dwi_patches = []
        t1_patches = []
        gt_dti_patches = []
        wm_mask_patches = []
        # dwi_unnorm_patches = []

        for patch_coord in sample.coords_data:
            x_start = patch_coord['x_start']
            x_end = patch_coord['x_end']
            y_start = patch_coord['y_start']
            y_end = patch_coord['y_end']
            z_start = patch_coord['z_start']
            z_end = patch_coord['z_end']

        
            dwi_patch = torch.from_numpy(sample.dwi[x_start:x_end, y_start:y_end, z_start:z_end])
            t1_patch = torch.from_numpy(sample.t1[x_start:x_end, y_start:y_end, z_start:z_end])
            gt_dti_patch = torch.from_numpy(sample.gt_dti[x_start:x_end, y_start:y_end, z_start:z_end])
            wm_mask_patch = torch.from_numpy(sample.wm_mask[x_start:x_end, y_start:y_end, z_start:z_end])
            # dwi_unnorm_patch = torch.from_numpy(sample.dwi_unnorm[x_start:x_end, y_start:y_end, z_start:z_end])

            dwi_patch = dwi_patch.permute(3,0,1,2)
            t1_patch = t1_patch.unsqueeze(0)
            gt_dti_patch = gt_dti_patch.permute(3,0,1,2)
            wm_mask_patch = wm_mask_patch.unsqueeze(0)
            # dwi_unnorm_patch = dwi_unnorm_patch.permute(3,0,1,2)

            dwi_patches.append(dwi_patch)
            t1_patches.append(t1_patch)
            gt_dti_patches.append(gt_dti_patch)
            wm_mask_patches.append(wm_mask_patch)
            # dwi_unnorm_patches.append(dwi_unnorm_patch)

        dwi_patches = torch.stack(dwi_patches, dim = 0)
        t1_patches = torch.stack(t1_patches, dim = 0)
        gt_dti_patches = torch.stack(gt_dti_patches, dim = 0)
        wm_mask_patches = torch.stack(wm_mask_patches, dim = 0)
        # dwi_unnorm_patches = torch.stack(dwi_unnorm_patches, dim = 0)

        # print('Shape of dwi_patch: {:}, gt_dti: {:}, t1: {:}, wm_mask : {:}'.format(dwi_patches.shape
        # ,gt_dti_patches.shape,t1_patches.shape, wm_mask_patches.shape))
        return {'dwi' : dwi_patches, 't1': t1_patches, 'gt_dti': gt_dti_patches, 'wm_mask': wm_mask_patches
            , 'grad': grad}
