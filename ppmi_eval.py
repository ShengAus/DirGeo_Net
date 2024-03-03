import os
import numpy as np

from utils.dmri_io import *
from utils.utils import *
from utils.metrics import *
import pickle

root = '/home/sheng/Diffusion/PPMI'
ppmi_split_path = os.path.join('/home/sheng/Diffusion/data', 'generated_results/sublist/PPMI_sub_list_updated.pickle')
testsubs = get_PPMIsamples(ppmi_split_path, train = False)
exp = 'PPMIFaSeUV2_100'
print(len(testsubs))

evals = {'dti':[],'fa':[],'md':[],'ad':[],'rd':[]}

for sub in testsubs:
    
    har_dti = load_data(os.path.join(root, 'sheng/DTI_RAS',sub , sub+'_dti_tensor.nii.gz'))
    pred_dti = load_data(os.path.join('/home/sheng/Diffusion/miccai_2024/results/', exp, sub + '_dti_pred.nii.gz'))
    wm_mask = load_data(os.path.join(root, 'sheng/wm_masks_RAS/', sub + '_wm_RAS.nii.gz'))

    har_fa, har_md, har_ad, har_rd = get_all_dti_metrics(har_dti)
    pred_fa, pred_md, pred_ad, pred_rd = get_all_dti_metrics(pred_dti)

    eval_dti = mae_metric(har_dti, pred_dti, wm_mask)
    eval_fa = mae_metric(har_fa, pred_fa, wm_mask)
    eval_md = mae_metric(har_md, pred_md, wm_mask)
    eval_ad = mae_metric(har_ad, pred_ad, wm_mask)
    eval_rd = mae_metric(har_rd, pred_rd, wm_mask)

    evals['dti'].append(eval_dti)
    evals['fa'].append(eval_fa)
    evals['md'].append(eval_md)
    evals['ad'].append(eval_ad)
    evals['rd'].append(eval_rd)

for k, v in evals.items():
    print(np.mean(evals[k]), np.std(evals[k]))

save_path = os.path.join('/home/sheng/Diffusion/miccai_2024/results/',exp, 'evals_mae.pickle')
with open(save_path, 'wb') as handle:
    pickle.dump(evals, handle, protocol=pickle.HIGHEST_PROTOCOL)





