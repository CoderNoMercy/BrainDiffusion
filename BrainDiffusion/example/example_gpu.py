'''
This example code present the basic use of package BrainDiffusion

The package should be imported as from BrainDiffusion import diffusion_brain_net
Here in the example, let's use the one example DTI data from HABS dataset. It will
be downloaded from the Google Drive into folder dti_path.

The output_path contain the solution of time dependent PDE in format .nii.gz
Connectivity matrix will be in the name of conn.mat under the folder output_path

Author: Zheyu Wen
Version: Jul. 17, 2023

'''

import os
from BrainDiffusion import BrainDiffusion_gpu
import scipy.io as sio
import numpy as np

# if use default dataset. This option will download data for you.
use_default_data=True

# the path to save DTI data
dti_path = os.path.dirname(os.path.realpath(__file__)) 

# the path to save Template (parcellation) of patient
template_file = os.path.dirname(os.path.realpath(__file__)) + '/data/P_BCNZ72/Template4_warped_labels.nii.gz'

# Patient ID of the DTI data you use
subj_name = 'P_BCNZ72'

# output_path save the PDE solution, and should be sufficient to contain 20GBs data
output_path = dti_path

# function to solve time-dependent PDE and save result in output_path
# Notice that diff_time=1 is only for a quick example. I would suggest diff_time=100 for sufficient diffusion.
BrainDiffusion_gpu.eval_diffusion(use_default_data, dti_path, template_file, subj_name, output_path, diff_time=100)

# construct connectivity matrix using the following function.
# The constructed matrix will be saved in conn_output_fname.
conn_output_fname = output_path + '/conn.mat'
BrainDiffusion_gpu.gen_conn_from_diffusion(dti_path, template_file, output_path, conn_output_fname)

# sanity check for the connectivity matrix. If use default dataset, the connecitivity matrix should has 
# shape (114, 114) and it should be a symmetric real matrix.
conn_mtx = np.asarray(sio.loadmat(conn_output_fname)['data']).astype('float')
print('connectivity matrix size', conn_mtx.shape)
print('Is it symmetric? ', np.array_equal(conn_mtx, conn_mtx.T))


