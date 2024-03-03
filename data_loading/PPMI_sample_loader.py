import os
import numpy as np
from data_loading.interfaces.angular_sr_nifti_io import AngularSRNiftiIO
from data_loading.angular_sr_sample import AnuglarSRSample


class PPMISampleLoader:

    def __init__(self, root, isTrain = True):
        self.root = root
        self.interface = AngularSRNiftiIO()
        self.isTrain = isTrain


    #---------------------------------------------#
    #                Load a sample                #
    #---------------------------------------------#
    # Load a sample from the data set
    def load_sample(self, uni, grad_dirs):
        # load data of a sample
        if self.isTrain:
            dwi_path = os.path.join(self.root, 'sheng/DWI_processed_RAS_mif',  uni + '_DWI_Denoised_Unringed_Upsampled.mif.gz')
            dwi, grad = self.load_training_dwi(path = dwi_path, index = uni, grad_dirs = grad_dirs, random = True)
        else: # testing
            dwi_path = os.path.join(self.root, 'sheng/DWI_processed_RAS_mif',  uni + '_DWI_Denoised_Unringed_Upsampled.mif.gz')
            dwi, grad = self.load_testing_dwiV1(path = dwi_path, index = uni, grad_dirs = grad_dirs, random = False)
        
        t1 =  self.load_t1(dir_name = 'sheng/T1_regrided', uni = uni,  suffix = '_T1_regrided.nii.gz')
        gt_dti, affine = self.load_gt(dir_name = 'sheng/DTI_RAS', uni = uni,  suffix = '_dti_tensor.nii.gz')
        brain_mask = self.load_brain_mask(dir_name = 'sheng/DWI_masks_RAS', uni = uni,  suffix = '_brainmask_RAS.nii.gz')
        wm_mask = self.load_wm_mask(dir_name = 'sheng/wm_masks_RAS', uni = uni,  suffix = '_wm_RAS.nii.gz')

        sample = AnuglarSRSample(uni, dwi, affine)
        sample.add_grad(grad)
        sample.add_gt_dti(gt_dti)
        sample.add_t1(t1)
        sample.add_brain_mask(brain_mask)
        sample.add_wm_mask(wm_mask)

        # if self.isTrain is False:
        #     gm_mask = self.load_gm_mask(dir_name = 'self/gm_mask', uni = uni,  suffix = '_gmmask.mif.gz')
        #     sample.add_gm_mask(gm_mask)
       
        return sample

    #---------------------------------------------#
    #                Load DWI                     #
    #---------------------------------------------#
    # Load the data of a sample from the data set
    def load_training_dwi(self, path, index, grad_dirs, random):
        ########
        # TBA  #
        ########
        dwi, lr_bvecs, lr_bvals = self.interface.load_downsampled_mif_dwi(path = path
        , index=index, grad_dirs = grad_dirs)

        # print(lr_bvecs.shape, lr_bvals.shape)

        grad = np.concatenate([lr_bvecs, np.expand_dims(lr_bvals, axis=1)], axis = -1)
        

        return dwi, grad

    def load_testing_dwiV1(self, path, index, grad_dirs, random):

        ########
        # TBA  #
        ########
        dwi, lr_bvecs, lr_bvals = self.interface.load_downsampled_mif_dwi(path = path
        , index=index, grad_dirs =grad_dirs, random=random )

        grad = np.concatenate([lr_bvecs, np.expand_dims(lr_bvals, axis=1)], axis = -1)

        return dwi, grad


    #---------------------------------------------#
    #                Load T1                      #
    #---------------------------------------------#
    # Load the data of a sample from the data set
    def load_t1(self, dir_name , uni , suffix):
        t1_path = os.path.join(self.root, dir_name,  uni + suffix)
        t1 = self.interface.load_data(t1_path)

        return t1

    #---------------------------------------------#
    #                Load GT                      #
    #---------------------------------------------#
    # Load the data of a sample from the data set
    def load_gt(self, dir_name , uni , suffix):
        gt_path = os.path.join(self.root, dir_name, uni, uni + suffix)
        gt, affine = self.interface.load_data(gt_path, needs_affine = True)

        return gt, affine

    #---------------------------------------------#
    #                Load Brain  mask             #
    #---------------------------------------------#
    def load_brain_mask(self, dir_name , uni , suffix):
        brain_mask_path = os.path.join(self.root, dir_name,  uni + suffix)
        brain_mask = self.interface.load_data(brain_mask_path)

        return brain_mask

    #---------------------------------------------#
    #                Load WM mask                 #
    #---------------------------------------------#
    def load_wm_mask(self, dir_name , uni , suffix):
        wm_mask_path = os.path.join(self.root, dir_name,  uni + suffix)
        wm_mask = self.interface.load_data(wm_mask_path)

        return wm_mask

    #---------------------------------------------#
    #                Load GM mask                 #
    #---------------------------------------------#
    def load_gm_mask(self, dir_name , uni , suffix):
        gm_mask_path = os.path.join(self.root, dir_name, uni + suffix)
        gm_mask = self.interface.load_data(gm_mask_path)

        return gm_mask

  
    def save_prediction(self, prediction, affine, output_name):
        self.interface.save_prediction(prediction, affine, output_name)

    





        



