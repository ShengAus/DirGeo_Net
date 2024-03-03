import os
import numpy as np
from data_loading.interfaces.angular_sr_nifti_io import AngularSRNiftiIO
from data_loading.angular_sr_sample import AnuglarSRSample


class HCPAnyDirsSampleLoader:

    def __init__(self, root, isTrain = True):
        self.root = root
        self.interface = AngularSRNiftiIO()
        self.isTrain = isTrain


    #---------------------------------------------#
    #                Load a sample                #
    #---------------------------------------------#
    # Load a sample from the data set
    def load_sample(self, uni, grad_dirs):
        if self.isTrain:
            dwi_path = os.path.join(self.root, 'dwi',  uni + '_DWI_processed_b1000.mif.gz')
            dwi, grad = self.load_training_dwi(path = dwi_path, index = uni, grad_dirs = grad_dirs, random = True)
        else: # testing
            dwi_path = os.path.join(self.root, 'dwi',  uni + '_DWI_processed_b1000.mif.gz')
            dwi, grad = self.load_testing_dwiV1(path = dwi_path, index = uni, grad_dirs = 6, random = False)
        
        gt_path = os.path.join(self.root, 'self/dti_nii_tensor_only', uni + '_DTI.nii.gz')
        gt_dti, affine = self.load_gt(path = gt_path, needs_affine = True)

        t1_path = os.path.join(self.root, 'self/T1_registered_RAS',  uni + '_t1.nii.gz')
        t1 =  self.load_t1(path = t1_path, needs_affine = False)

        brain_mask_path = os.path.join(self.root, 'self/mask_RAS',  uni + '_DWI_brainmask.mif.gz')
        brain_mask = self.load_brain_mask(path =brain_mask_path, needs_affine = False)

        wm_mask_path = os.path.join(self.root, 'self/wm_mask_RAS',  uni + '_wm.mif.gz')
        wm_mask = self.load_wm_mask(path = wm_mask_path, needs_affine = False)
  

        sample = AnuglarSRSample(uni, dwi, affine)
        sample.add_grad(grad)
        sample.add_gt_dti(gt_dti)
        sample.add_t1(t1)
        sample.add_brain_mask(brain_mask)
        sample.add_wm_mask(wm_mask)
       
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
    def load_t1(self, path, needs_affine = False):
        t1 = self.interface.load_data(path, needs_affine = needs_affine)

        return t1

    #---------------------------------------------#
    #                Load GT                      #
    #---------------------------------------------#
    # Load the data of a sample from the data set
    def load_gt(self, path, needs_affine = True):
        gt, affine = self.interface.load_data(path, needs_affine = needs_affine)

        return gt, affine

    #---------------------------------------------#
    #                Load Brain  mask             #
    #---------------------------------------------#
    def load_brain_mask(self, path, needs_affine = False):
        brain_mask = self.interface.load_data(path, needs_affine = needs_affine)

        return brain_mask

    #---------------------------------------------#
    #                Load WM mask                 #
    #---------------------------------------------#
    def load_wm_mask(self, path, needs_affine = False):
        wm_mask = self.interface.load_data(path, needs_affine = False)

        return wm_mask

    #---------------------------------------------#
    #                Load GM mask                 #
    #---------------------------------------------#
    def load_gm_mask(self, path, needs_affine = False):
        gm_mask = self.interface.load_data(path, needs_affine = False)

        return gm_mask


    def save_prediction(self, prediction, affine, output_name):
        self.interface.save_prediction(prediction, affine, output_name)

    





        



