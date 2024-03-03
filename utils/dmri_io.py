from utils.mrtrix import *
import os
import nibabel as nib
import numpy as np
import pickle

def get_HCPsamples(hcp_split_path, train =True):
    if not os.path.exists(hcp_split_path):
        raise IOError(
            "hcp splited list path, {}, could not be resolved".format(hcp_split_path)
        )
        exit(0)

    with open(hcp_split_path, 'rb') as handle:
        sub_list = pickle.load(handle)
        
        if train:
            sample_list = sub_list['train']
        else:
            sample_list = sub_list['test']
            
    return sample_list

def get_PPMIsamples(ppmi_split_path, train=True):
    if not os.path.exists(ppmi_split_path):
        raise IOError(
            "PPMI splited list path, {}, could not be resolved".format(ppmi_split_path)
        )
        exit(0)
    
    with open(ppmi_split_path, 'rb') as handle:
        sub_list = pickle.load(handle)

        sample_list = []
        if train:
            sample_list.extend(sub_list['CONTROL_train'])
            sample_list.extend(sub_list['PD_train'])
            sample_list.remove('40541')
        else:
            sample_list.extend(sub_list['CONTROL_test'])
            sample_list.extend(sub_list['PD_test'])

    return sample_list

def get_PPMISelectedsamples(ppmi_split_path, train=True):
    if not os.path.exists(ppmi_split_path):
        raise IOError(
            "PPMI splited list path, {}, could not be resolved".format(ppmi_split_path)
        )
        exit(0)
    
    with open(ppmi_split_path, 'rb') as handle:
        sub_list = pickle.load(handle)

        sample_list = []
        if train:
            ratio = len(sub_list['CONTROL_train'])/len(sub_list['PD_train'])
            x = int(80/(ratio+1)*ratio)
            sample_list.extend(sub_list['CONTROL_train'][0:x])
            sample_list.extend(sub_list['PD_train'][0:80-x])
            if '40541' in sample_list:
                sample_list.remove('40541')
        else:
            ratio = len(sub_list['CONTROL_test'])/len(sub_list['PD_test'])
            x = int(20/(ratio+1)*ratio)
            sample_list.extend(sub_list['CONTROL_test'][0:x])
            sample_list.extend(sub_list['PD_test'][0:80-x])

    return sample_list

def load_data(path, needs_affine = False):

    if not os.path.exists(path):
        raise ValueError(
            "Data could not be found \"{}\"".format(path)
        )
        exit(0)

    if path.endswith('.mif.gz') or path.endswith('.mif'):
        vol = load_mrtrix(path)
        data_copied = vol.data.copy()
        affine_copied = vol.transform.copy()
    elif path.endswith('.nii.gz') or path.endswith('.nii'):
        vol = nib.load(path)
        data_copied = vol.get_fdata().copy()
        affine_copied = vol.affine.copy()
    else:
        raise IOError('file extension not supported: ' + str(path))
        exit(0)

    # Return volume
    if needs_affine:
        return data_copied, affine_copied
    else:
        return data_copied